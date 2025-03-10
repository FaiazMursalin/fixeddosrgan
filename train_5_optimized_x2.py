import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from helpers.utils_2_x2 import SRDataset, evaluate_model, save_results
from model_5_x2 import ProposedGenerator, ContentLoss
from tqdm import tqdm


def check_loss_value(loss, name, threshold=5.0):
    if torch.isnan(loss).any():
        print(f"Warning: NaN detected in {name}")
        return torch.tensor(0.0).to(loss.device)
    if loss.item() > threshold:
        print(f"Warning: {name} exceeded threshold ({loss.item():.4f})")
        return torch.tensor(threshold).to(loss.device)
    return loss


def train_model(generator, train_loader, val_loader, test_set5_loader, test_set14_loader,
                num_epochs, device, scale_factor):
    scaler = GradScaler()

    # Loss functions
    content_criterion = ContentLoss().to(device)
    criterion_pixel = nn.L1Loss().to(device)

    # Optimizer - Updated learning rate and parameters for Swin Transformer
    optimizer_G = torch.optim.AdamW(
        generator.parameters(),
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01  # Added weight decay for regularization
    )

    # Learning rate scheduler - Updated for transformer-based model
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(  # Changed to cosine annealing
        optimizer_G,
        T_0=50,  # Restart every 50 epochs
        T_mult=2,  # Double the restart interval each time
        eta_min=1e-7
    )

    # Dynamic loss weights function - Adjusted phase timing
    def get_loss_weights(epoch, max_epochs):
        if epoch < max_epochs * 0.3:  # First 30% epochs
            return 1.0, 0.0  # Only pixel loss
        else:  # Last 70% epochs
            return 0.85, 0.15  # Adjusted weights for pixel + content loss

    # Rest of the configuration
    history = {
        'train_loss': [],
        'val_psnr': [], 'val_ssim': [],
        'set5_psnr': [], 'set5_ssim': [],
        'set14_psnr': [], 'set14_ssim': []
    }

    model_save_path = f'proposed_model_swin_x{scale_factor}.pth'
    best_model_path = f'best_model_swin_x{scale_factor}.pth'
    accumulation_steps = 4
    best_psnr = 0

    print("Starting training with phased approach:")
    print("Phase 1 (0-30%): Pure Pixel Loss")
    print("Phase 2 (30-100%): Pixel + Content Loss")

    for epoch in range(num_epochs):
        generator.train()
        total_loss = 0
        batch_count = 0

        lambda_pixel, lambda_content = get_loss_weights(epoch, num_epochs)

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        optimizer_G.zero_grad()

        for lr_imgs, hr_imgs in progress_bar:
            batch_count += 1
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            with autocast():
                sr_imgs = generator(lr_imgs)
                sr_imgs = torch.clamp(sr_imgs, 0, 1)

                pixel_loss = criterion_pixel(sr_imgs, hr_imgs)
                pixel_loss = torch.clamp(pixel_loss, 0, 5.0)
                loss = lambda_pixel * pixel_loss

                if epoch >= num_epochs * 0.3:  # Updated timing
                    content_loss = content_criterion(sr_imgs, hr_imgs)
                    content_loss = torch.clamp(content_loss, 0, 5.0)
                    loss = loss + lambda_content * content_loss
                else:
                    content_loss = torch.tensor(0.0, device=device)

            scaler.scale(loss / accumulation_steps).backward()

            # Added gradient clipping specifically for transformer
            if batch_count % accumulation_steps == 0:
                scaler.unscale_(optimizer_G)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)  # Increased max_norm
                scaler.step(optimizer_G)
                optimizer_G.zero_grad()
                scaler.update()

            total_loss += loss.item()

            current_lr = optimizer_G.param_groups[0]['lr']

            # Update progress bar
            phase = "Pixel" if epoch < num_epochs * 0.3 else "Pixel+Content"
            progress_bar.set_postfix({
                'Phase': phase,
                'Loss': f'{loss.item():.4f}',
                'Pixel': f'{pixel_loss.item():.4f}',
                'Content': f'{content_loss.item():.4f}',
                'LR': f'{current_lr:.2e}'
            })

        # Calculate average epoch loss
        avg_loss = total_loss / len(train_loader)

        # Evaluate model
        generator.eval()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                val_metrics = evaluate_model(generator, val_loader, device, "Validation")
                set5_metrics = evaluate_model(generator, test_set5_loader, device, "Set5")
                set14_metrics = evaluate_model(generator, test_set14_loader, device, "Set14")
        generator.train()

        # Save best model based on Set5 PSNR
        if set5_metrics[0] > best_psnr:
            best_psnr = set5_metrics[0]
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'best_psnr': best_psnr,
            }, best_model_path)
            print(f"\nNew best model saved! Set5 PSNR: {best_psnr:.2f}")

        # Update learning rate
        scheduler_G.step(set5_metrics[0])

        # Update history
        history['train_loss'].append(avg_loss)
        history['val_psnr'].append(val_metrics[0])
        history['val_ssim'].append(val_metrics[1])
        history['set5_psnr'].append(set5_metrics[0])
        history['set5_ssim'].append(set5_metrics[1])
        history['set14_psnr'].append(set14_metrics[0])
        history['set14_ssim'].append(set14_metrics[1])

        # Save latest model
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'scaler': scaler.state_dict(),
            'psnr_set5': set5_metrics[0],
            'ssim_set5': set5_metrics[1],
            'psnr_set14': set14_metrics[0],
            'ssim_set14': set14_metrics[1],
            'loss': avg_loss,
            'history': history
        }, model_save_path)

        print(f'\nEpoch {epoch + 1} Results:')
        print(f'Phase: {phase}')
        print(f'Loss: {avg_loss:.4f}')
        print(f'Set5 - PSNR: {set5_metrics[0]:.2f}, SSIM: {set5_metrics[1]:.4f}')
        print(f'Set14 - PSNR: {set14_metrics[0]:.2f}, SSIM: {set14_metrics[1]:.4f}')
        print(f'Learning Rate: {current_lr:.2e}\n')

    return history


if __name__ == "__main__":
    try:
        # Configuration
        scale_factor = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 16  # change it to 16
        num_epochs = 1000

        torch.cuda.empty_cache()

        print(f"Using device: {device}")
        print(f"Scale factor: {scale_factor}x")
        print(f"Batch size: {batch_size}")
        print(f"Effective batch size: {batch_size * 4}")  # Due to gradient accumulation
        print(f"Number of epochs: {num_epochs}")

        # Initialize generator only
        generator = ProposedGenerator(scale_factor=scale_factor).to(device)

        # Set to half precision to save memory
        # if torch.cuda.is_available():
        #     generator = generator.half()
        #     print("Using half precision (FP16) to save memory")

        # Dataset paths
        general100_hr_dir = './dataset_improved/general100/x2/hr'
        general100_lr_dir = './dataset_improved/general100/x2/lr'
        set5_hr_dir = './dataset_improved/set5/x2/hr'
        set5_lr_dir = './dataset_improved/set5/x2/lr'
        set14_hr_dir = './dataset_improved/set14/x2/hr'
        set14_lr_dir = './dataset_improved/set14/x2/lr'

        # Create datasets
        general100_dataset = SRDataset(general100_hr_dir, general100_lr_dir)
        set5_dataset = SRDataset(set5_hr_dir, set5_lr_dir)
        set14_dataset = SRDataset(set14_hr_dir, set14_lr_dir)

        # Split General100 into train/val
        train_size = int(0.8 * len(general100_dataset))
        val_size = len(general100_dataset) - train_size
        train_dataset, val_dataset = random_split(
            general100_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=4
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_set5_loader = DataLoader(set5_dataset, batch_size=1, shuffle=False)
        test_set14_loader = DataLoader(set14_dataset, batch_size=1, shuffle=False)

        print("\nDataset sizes:")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Set5 test samples: {len(set5_dataset)}")
        print(f"Set14 test samples: {len(set14_dataset)}")

        print("\nTraining phases:")
        print("Phase 1 (0-50%): Pixel Loss Only")
        print("Phase 2 (50-100%): Pixel + Content Loss")

        # Train the model
        history = train_model(
            generator=generator,
            train_loader=train_loader,
            val_loader=val_loader,
            test_set5_loader=test_set5_loader,
            test_set14_loader=test_set14_loader,
            num_epochs=num_epochs,
            device=device,
            scale_factor=scale_factor
        )

        # Plot and save results
        # plot_training_curves(history)
        # print("Training curves plotted")

        # Final evaluation
        print("\nPerforming final evaluation...")
        generator.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                set5_psnr, set5_ssim = evaluate_model(generator, test_set5_loader, device, "Set5")
                set14_psnr, set14_ssim = evaluate_model(generator, test_set14_loader, device, "Set14")

        final_metrics = {
            'Set5': {'psnr': set5_psnr, 'ssim': set5_ssim},
            'Set14': {'psnr': set14_psnr, 'ssim': set14_ssim}
        }

        save_results(history, final_metrics)
        print("\nFinal Results:")
        print(f"Set5 - PSNR: {set5_psnr:.2f}, SSIM: {set5_ssim:.4f}")
        print(f"Set14 - PSNR: {set14_psnr:.2f}, SSIM: {set14_ssim:.4f}")
    except OSError as error:
        print(error)