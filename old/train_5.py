import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils import SRDataset, evaluate_model, plot_training_curves, save_results
from old.model_5 import ProposedGenerator, ProposedDiscriminator, ContentLoss
from tqdm import tqdm

def train_model(generator, discriminator, train_loader, val_loader, test_set5_loader, test_set14_loader, 
                num_epochs, device, scale_factor):
    
    # Loss functions
    content_criterion = ContentLoss().to(device)
    adversarial_criterion = nn.BCELoss()

    # Optimizers with AdamW and weight decay
    optimizer_G = optim.AdamW(generator.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=1e-4)

    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.OneCycleLR(
        optimizer_G, max_lr=1e-4, epochs=num_epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    scheduler_D = optim.lr_scheduler.OneCycleLR(
        optimizer_D, max_lr=5e-5, epochs=num_epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    history = {
        'train_loss_G': [], 'train_loss_D': [],
        'val_psnr': [], 'val_ssim': [],
        'set5_psnr': [], 'set5_ssim': [],
        'set14_psnr': [], 'set14_ssim': []
    }

    best_psnr = 0
    model_save_path = f'proposed_model_x{scale_factor}.pth'

    print("Starting training...")
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        total_g_loss = 0
        total_d_loss = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for lr_imgs, hr_imgs in progress_bar:
            batch_size = lr_imgs.size(0)
            real_label = torch.ones(batch_size, device=device)
            fake_label = torch.zeros(batch_size, device=device)
            
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # Train Generator
            optimizer_G.zero_grad()
            sr_imgs = generator(lr_imgs)
            
            # Calculate generator losses
            content_loss = content_criterion(sr_imgs, hr_imgs)
            fake_output = discriminator(sr_imgs)
            adversarial_loss = adversarial_criterion(fake_output, real_label)
            
            g_loss = content_loss + 0.001 * adversarial_loss
            
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()
            
            # Train Discriminator (every other batch)
            if batch_size % 2 == 0:
                optimizer_D.zero_grad()
                
                real_output = discriminator(hr_imgs)
                d_loss_real = adversarial_criterion(real_output, real_label)
                
                fake_output = discriminator(sr_imgs.detach())
                d_loss_fake = adversarial_criterion(fake_output, fake_label)
                
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                optimizer_D.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            progress_bar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })

            scheduler_G.step()
            scheduler_D.step()

        # Calculate average epoch losses
        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)

        # Evaluate on validation and test sets
        val_metrics = evaluate_model(generator, val_loader, device, "Validation")
        set5_metrics = evaluate_model(generator, test_set5_loader, device, "Set5")
        set14_metrics = evaluate_model(generator, test_set14_loader, device, "Set14")

        # Update history
        history['train_loss_G'].append(avg_g_loss)
        history['train_loss_D'].append(avg_d_loss)
        history['val_psnr'].append(val_metrics[0])
        history['val_ssim'].append(val_metrics[1])
        history['set5_psnr'].append(set5_metrics[0])
        history['set5_ssim'].append(set5_metrics[1])
        history['set14_psnr'].append(set14_metrics[0])
        history['set14_ssim'].append(set14_metrics[1])

        # Save best model
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'psnr_set5': set5_metrics[0],
            'ssim_set5': set5_metrics[1],
            'psnr_set14': set14_metrics[0],
            'ssim_set14': set14_metrics[1],
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
        }, model_save_path)
        print(f'Saved model at epoch {epoch+1} with Set5 PSNR: {set5_metrics[0]:.2f}, Set14 PSNR: {set14_metrics[0]:.2f}')

    return history

if __name__ == "__main__":
    # Configuration
    scale_factor = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    num_epochs = 100

    print(f"Using device: {device}")
    print(f"Scale factor: {scale_factor}x")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")

    # Initialize models
    generator = ProposedGenerator(scale_factor=scale_factor).to(device)
    discriminator = ProposedDiscriminator().to(device)

    # Dataset paths
    general100_hr_dir = './dataset_improved/general100/x4/hr'
    general100_lr_dir = './dataset_improved/general100/x4/lr'
    set5_hr_dir = './dataset_improved/set5/x4/hr'
    set5_lr_dir = './dataset_improved/set5/x4/lr'
    set14_hr_dir = './dataset_improved/set14/x4/hr'
    set14_lr_dir = './dataset_improved/set14/x4/lr'

    # Create datasets
    general100_dataset = SRDataset(general100_hr_dir, general100_lr_dir)
    set5_dataset = SRDataset(set5_hr_dir, set5_lr_dir)
    set14_dataset = SRDataset(set14_hr_dir, set14_lr_dir)

    # Split General100 into train/val
    train_size = int(0.8 * len(general100_dataset))
    val_size = len(general100_dataset) - train_size
    train_dataset, val_dataset = random_split(general100_dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_set5_loader = DataLoader(set5_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_set14_loader = DataLoader(set14_dataset, batch_size=1, shuffle=False, num_workers=4)

    print(f"Dataset sizes:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Set5 test samples: {len(set5_dataset)}")
    print(f"Set14 test samples: {len(set14_dataset)}")

    # Train the model
    history = train_model(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        test_set5_loader=test_set5_loader,
        test_set14_loader=test_set14_loader,
        num_epochs=num_epochs,
        device=device,
        scale_factor=scale_factor
    )

    # Plot training curves
    plot_training_curves(history)
    print("Training curves plotted")

    # Final evaluation
    print("\nPerforming final evaluation...")
    
    # Evaluate on test sets
    set5_psnr, set5_ssim = evaluate_model(generator, test_set5_loader, device, "Set5")
    set14_psnr, set14_ssim = evaluate_model(generator, test_set14_loader, device, "Set14")

    # Compile final metrics
    final_metrics = {
        'Set5': {
            'psnr': set5_psnr,
            'ssim': set5_ssim
        },
        'Set14': {
            'psnr': set14_psnr,
            'ssim': set14_ssim
        }
    }

    # Save final results
    save_results(history, final_metrics)
    print("Results saved")

    print("\nTraining completed!")
    print("\nFinal Results:")
    print(f"Set5 - PSNR: {set5_psnr:.2f}, SSIM: {set5_ssim:.4f}")
    print(f"Set14 - PSNR: {set14_psnr:.2f}, SSIM: {set14_ssim:.4f}")
