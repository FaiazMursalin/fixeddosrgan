import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from old.models import ImprovedSRCNN, Discriminator, ContentLoss


class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.image_files = [f for f in os.listdir(lr_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        lr_img = Image.open(self.lr_dir / img_name).convert('RGB')
        hr_img = Image.open(self.hr_dir / img_name).convert('RGB')
        transform = transforms.ToTensor()
        return transform(lr_img), transform(hr_img)


def gaussian_kernel(size=11, sigma=1.5):
    x = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    gauss = np.exp(-0.5 * np.square(x / sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / kernel.sum()


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)


def calculate_ssim(img1, img2, window_size=11):
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    kernel = gaussian_kernel(window_size)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(img1.device)
    kernel = kernel.expand(3, 1, window_size, window_size)

    mu1 = nn.functional.conv2d(img1, kernel, padding=window_size // 2, groups=3)
    mu2 = nn.functional.conv2d(img2, kernel, padding=window_size // 2, groups=3)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = nn.functional.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=3) - mu1_sq
    sigma2_sq = nn.functional.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=3) - mu2_sq
    sigma12 = nn.functional.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=3) - mu12

    ssim = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
           ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim.mean()


def calculate_ifc(img1, img2, window_size=11):
    img1_y = 0.299 * img1[0] + 0.587 * img1[1] + 0.114 * img1[2]
    img2_y = 0.299 * img2[0] + 0.587 * img2[1] + 0.114 * img2[2]

    kernel = torch.ones(1, 1, window_size, window_size).to(img1.device) / (window_size ** 2)

    mu1 = nn.functional.conv2d(img1_y.unsqueeze(0).unsqueeze(0), kernel, padding=window_size // 2)
    mu2 = nn.functional.conv2d(img2_y.unsqueeze(0).unsqueeze(0), kernel, padding=window_size // 2)

    sigma1_sq = nn.functional.conv2d(img1_y.unsqueeze(0).unsqueeze(0) ** 2, kernel, padding=window_size // 2) - mu1 ** 2
    sigma2_sq = nn.functional.conv2d(img2_y.unsqueeze(0).unsqueeze(0) ** 2, kernel, padding=window_size // 2) - mu2 ** 2
    sigma12 = nn.functional.conv2d((img1_y * img2_y).unsqueeze(0).unsqueeze(0), kernel,
                                   padding=window_size // 2) - mu1 * mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = torch.log2(1 + (sigma12 ** 2 + C2) / (sigma1_sq * sigma2_sq + C1))

    return numerator.mean()


def evaluate_model(model, test_loader, device, dataset_name):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_ifc = 0

    print(f"\nEvaluating on {dataset_name}...")
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(test_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            sr_imgs = model(lr_imgs)

            for i in range(sr_imgs.size(0)):
                psnr = calculate_psnr(sr_imgs[i], hr_imgs[i])
                ssim = calculate_ssim(sr_imgs[i], hr_imgs[i])
                ifc = calculate_ifc(sr_imgs[i], hr_imgs[i])

                total_psnr += psnr.cpu().item()
                total_ssim += ssim.cpu().item()
                total_ifc += ifc.cpu().item()

    num_images = len(test_loader.dataset)
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    avg_ifc = total_ifc / num_images

    print(f"\nResults on {dataset_name}:")
    print(f"PSNR: {avg_psnr:.2f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"IFC: {avg_ifc:.4f}")

    return avg_psnr, avg_ssim, avg_ifc


def train_gan_model(generator, discriminator, train_loader, val_loader, test_set5_loader, test_set14_loader, num_epochs,
                    device, scale_factor):
    content_criterion = ContentLoss().to(device)
    adversarial_criterion = nn.BCELoss()

    optimizer_G = optim.AdamW(generator.parameters(), lr=0.0002, betas=(0.9, 0.999), weight_decay=0.005)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)

    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=num_epochs, eta_min=1e-7)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=num_epochs, eta_min=1e-7)

    history = {
        'train_loss_G': [], 'train_loss_D': [],
        'val_psnr': [], 'val_ssim': [], 'val_ifc': [],
        'set5_psnr': [], 'set5_ssim': [], 'set5_ifc': [],
        'set14_psnr': [], 'set14_ssim': [], 'set14_ifc': []
    }

    best_psnr = 0
    model_save_path = f'improved_srcnn_gan_x{scale_factor}.pth'

    print("Training started...")
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        total_train_loss_G = 0
        total_train_loss_D = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for lr_imgs, hr_imgs in progress_bar:
            batch_size = lr_imgs.size(0)
            real_label = torch.ones(batch_size, device=device)
            fake_label = torch.zeros(batch_size, device=device)
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_output = discriminator(hr_imgs)
            d_loss_real = adversarial_criterion(real_output, real_label)

            with torch.no_grad():
                sr_imgs = generator(lr_imgs)
            fake_output = discriminator(sr_imgs.detach())
            d_loss_fake = adversarial_criterion(fake_output, fake_label)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            sr_imgs = generator(lr_imgs)
            fake_output = discriminator(sr_imgs)

            content_loss = content_criterion(sr_imgs, hr_imgs)
            adversarial_loss = adversarial_criterion(fake_output, real_label)
            g_loss = content_loss + 0.0001 * adversarial_loss

            g_loss.backward()
            optimizer_G.step()

            total_train_loss_G += g_loss.item()
            total_train_loss_D += d_loss.item()
            progress_bar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })

        avg_train_loss_G = total_train_loss_G / len(train_loader)
        avg_train_loss_D = total_train_loss_D / len(train_loader)

        # Evaluate on validation set
        val_psnr, val_ssim, val_ifc = evaluate_model(generator, val_loader, device, "Validation")

        # Evaluate on Set5
        set5_psnr, set5_ssim, set5_ifc = evaluate_model(generator, test_set5_loader, device, "Set5")

        # Evaluate on Set14
        set14_psnr, set14_ssim, set14_ifc = evaluate_model(generator, test_set14_loader, device, "Set14")

        # Update history
        history['train_loss_G'].append(avg_train_loss_G)
        history['train_loss_D'].append(avg_train_loss_D)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['val_ifc'].append(val_ifc)
        history['set5_psnr'].append(set5_psnr)
        history['set5_ssim'].append(set5_ssim)
        history['set5_ifc'].append(set5_ifc)
        history['set14_psnr'].append(set14_psnr)
        history['set14_ssim'].append(set14_ssim)
        history['set14_ifc'].append(set14_ifc)

        print(f'\nEpoch {epoch + 1} Results:')
        print(f'Train G Loss: {avg_train_loss_G:.4f}, D Loss: {avg_train_loss_D:.4f}')
        print(f'Validation - PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}, IFC: {val_ifc:.4f}')
        print(f'Set5 - PSNR: {set5_psnr:.2f}, SSIM: {set5_ssim:.4f}, IFC: {set5_ifc:.4f}')
        print(f'Set14 - PSNR: {set14_psnr:.2f}, SSIM: {set14_ssim:.4f}, IFC: {set14_ifc:.4f}')

        if set5_psnr > best_psnr:
            best_psnr = set5_psnr
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'best_psnr': best_psnr,
            }, model_save_path)
            print(f'Saved best model with Set5 PSNR: {best_psnr:.2f}')

        scheduler_G.step()
        scheduler_D.step()

    return history


def plot_training_curves(history):
    epochs = range(1, len(history['train_loss_G']) + 1)

    plt.figure(figsize=(20, 5))

    # Generator and Discriminator Loss
    plt.subplot(1, 4, 1)
    plt.plot(epochs, history['train_loss_G'], 'b-', label='Generator Loss')
    plt.plot(epochs, history['train_loss_D'], 'r-', label='Discriminator Loss')
    plt.title('Generator and Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # PSNR
    plt.subplot(1, 4, 2)
    plt.plot(epochs, history['val_psnr'], 'm-')
    plt.title('PSNR on Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')

    # SSIM
    plt.subplot(1, 4, 3)
    plt.plot(epochs, history['val_ssim'], 'y-')
    plt.title('SSIM on Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')

    # IFC
    plt.subplot(1, 4, 4)
    plt.plot(epochs, history['val_ifc'], 'c-')
    plt.title('IFC on Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('IFC')

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()


if __name__ == "__main__":
    # Configuration
    scale_factor = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 500

    print(f"Using device: {device}")
    print(f"Training configuration:")
    print(f"- Scale factor: {scale_factor}x")
    print(f"- Batch size: {batch_size}")
    print(f"- Number of epochs: {num_epochs}")

    # Create models
    generator = ImprovedSRCNN(scale_factor=scale_factor).to(device)
    discriminator = Discriminator().to(device)

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

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Set5 test samples: {len(set5_dataset)}")
    print(f"Set14 test samples: {len(set14_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_set5_loader = DataLoader(set5_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_set14_loader = DataLoader(set14_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Train the model
    history = train_gan_model(
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

    # Final evaluation on test sets
    print("\nFinal Evaluation:")
    print("\nEvaluating on Set5:")
    set5_metrics = evaluate_model(generator, test_set5_loader, device, "Set5")

    print("\nEvaluating on Set14:")
    set14_metrics = evaluate_model(generator, test_set14_loader, device, "Set14")

    # Save final metrics
    final_results = {
        'Set5': {
            'PSNR': set5_metrics[0],
            'SSIM': set5_metrics[1],
            'IFC': set5_metrics[2]
        },
        'Set14': {
            'PSNR': set14_metrics[0],
            'SSIM': set14_metrics[1],
            'IFC': set14_metrics[2]
        }
    }
    print(final_results)

    # Save results to file
    with open('old/final_results_x4.txt', 'w') as f:
        f.write("Final Evaluation Results\n")
        f.write("=======================\n\n")
        for dataset, metrics in final_results.items():
            f.write(f"{dataset} Results:\n")
            f.write(f"PSNR: {metrics['PSNR']:.2f}\n")
            f.write(f"SSIM: {metrics['SSIM']:.4f}\n")
            f.write(f"IFC: {metrics['IFC']:.4f}\n\n")

    print("\nTraining and evaluation completed!")
