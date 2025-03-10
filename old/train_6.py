import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils_2 import SRDataset, evaluate_model, plot_training_curves, save_results
from old.model_6 import ProposedGenerator, ContentLoss
from tqdm import tqdm

def train_model(generator, train_loader, val_loader, test_set5_loader, test_set14_loader, num_epochs, device):
    criterion_pixel = nn.L1Loss().to(device)
    criterion_content = ContentLoss().to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs, eta_min=1e-6)

    history = {
        'train_loss_G': [],
        'val_psnr': [], 'val_ssim': [],
        'set5_psnr': [], 'set5_ssim': [],
        'set14_psnr': [], 'set14_ssim': []
    }

    model_save_path = 'old/proposed_model_optimized.pth'
    best_model_path = 'old/best_model.pth'
    best_psnr = 0

    for epoch in range(num_epochs):
        generator.train()
        total_g_loss = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for lr_imgs, hr_imgs in progress_bar:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer_G.zero_grad()

            sr_imgs = generator(lr_imgs)

            pixel_loss = criterion_pixel(sr_imgs, hr_imgs)
            content_loss = criterion_content(sr_imgs, hr_imgs)

            g_loss = pixel_loss + 0.01 * content_loss

            g_loss.backward()
            optimizer_G.step()

            total_g_loss += g_loss.item()

            progress_bar.set_postfix({'Loss G': g_loss.item()})

        scheduler_G.step()

        avg_g_loss = total_g_loss / len(train_loader)

        generator.eval()
        val_metrics = evaluate_model(generator, val_loader, device, "Validation")
        set5_metrics = evaluate_model(generator, test_set5_loader, device, "Set5")
        set14_metrics = evaluate_model(generator, test_set14_loader, device, "Set14")
        generator.train()

        if set5_metrics[0] > best_psnr:
            best_psnr = set5_metrics[0]
            torch.save(generator.state_dict(), best_model_path)

        history['train_loss_G'].append(avg_g_loss)
        history['val_psnr'].append(val_metrics[0])
        history['val_ssim'].append(val_metrics[1])
        history['set5_psnr'].append(set5_metrics[0])
        history['set5_ssim'].append(set5_metrics[1])
        history['set14_psnr'].append(set14_metrics[0])
        history['set14_ssim'].append(set14_metrics[1])

        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'psnr_set5': set5_metrics[0],
            'ssim_set5': set5_metrics[1],
            'psnr_set14': set14_metrics[0],
            'ssim_set14': set14_metrics[1],
            'g_loss': avg_g_loss,
            'history': history
        }, model_save_path)

        print(f'\nEpoch {epoch+1} Results:')
        print(f'Loss G: {avg_g_loss:.4f}')
        print(f'Set5 - PSNR: {set5_metrics[0]:.2f}, SSIM: {set5_metrics[1]:.4f}')
        print(f'Set14 - PSNR: {set14_metrics[0]:.2f}, SSIM: {set14_metrics[1]:.4f}')

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

    # Initialize model
    generator = ProposedGenerator(scale_factor=scale_factor).to(device)

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

    print("\nTraining with pixel and content loss:")

    # Train the model
    history = train_model(
        generator=generator,
        train_loader=train_loader,
        val_loader=val_loader,
        test_set5_loader=test_set5_loader,
        test_set14_loader=test_set14_loader,
        num_epochs=num_epochs,
        device=device
    )

    # Plot and save results
    plot_training_curves(history)
    print("Training curves plotted")

    # Final evaluation
    print("\nPerforming final evaluation...")
    generator.eval()
    with torch.no_grad():
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
