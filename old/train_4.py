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
from tqdm import tqdm
import random
from old.models_4 import ImprovedSRCNN, Discriminator, ContentLoss


class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.image_files = [f for f in os.listdir(lr_dir) if f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((0, 270))  # Fixed: using tuple instead of list
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        lr_img = Image.open(self.lr_dir / img_name).convert('RGB')
        hr_img = Image.open(self.hr_dir / img_name).convert('RGB')

        # Apply same augmentation to both images
        if random.random() > 0.5:
            seed = random.randint(0, 2 ** 32)  # Set same seed for both transforms
            torch.manual_seed(seed)  # Added for better reproducibility
            random.seed(seed)
            lr_img = self.transform(lr_img)
            torch.manual_seed(seed)  # Added for better reproducibility
            random.seed(seed)
            hr_img = self.transform(hr_img)

        return transforms.ToTensor()(lr_img), transforms.ToTensor()(hr_img)


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
    num_images = 0

    print(f"\nEvaluating on {dataset_name}...")
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(test_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            sr_imgs = model(lr_imgs)

            batch_size = sr_imgs.size(0)
            num_images += batch_size

            for i in range(batch_size):
                psnr = calculate_psnr(sr_imgs[i], hr_imgs[i])
                ssim = calculate_ssim(sr_imgs[i], hr_imgs[i])
                ifc = calculate_ifc(sr_imgs[i], hr_imgs[i])

                total_psnr += psnr.cpu().item()
                total_ssim += ssim.cpu().item()
                total_ifc += ifc.cpu().item()

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

    # Adjusted learning rates for better stability
    optimizer_G = optim.AdamW(generator.parameters(), lr=0.0002, betas=(0.9, 0.999), weight_decay=0.0001)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=0.00004, betas=(0.9, 0.999), weight_decay=0.0001)

    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    scheduler_G = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_G,
        max_lr=0.0002,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=25,
        final_div_factor=1e4
    )

    scheduler_D = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_D,
        max_lr=0.00004,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=25,
        final_div_factor=1e4
    )

    history = {
        'train_loss_G': [], 'train_loss_D': [],
        'val_psnr': [], 'val_ssim': [], 'val_ifc': [],
        'set5_psnr': [], 'set5_ssim': [], 'set5_ifc': [],
        'set14_psnr': [], 'set14_ssim': [], 'set14_ifc': []
    }

    best_psnr = 0
    model_save_path = f'improved_srcnn_gan_x{scale_factor}_2.pth'

    print("Training started...")
    warmup_epochs = 5

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        total_train_loss_G = 0
        total_train_loss_D = 0

        # Warmup learning rate
        if epoch < warmup_epochs:
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = optimizer_G.param_groups[0]['initial_lr'] * (epoch + 1) / warmup_epochs

        # Set phase-based training
        if epoch < 100:
            adv_weight = 0  # Pure content loss
        elif epoch < 300:
            adv_weight = 0.00001  # Small adversarial component
        else:
            adv_weight = 0.00005  # Slightly increased adversarial

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

            content_loss = content_criterion(sr_imgs, hr_imgs)
            fake_output = discriminator(sr_imgs)
            adversarial_loss = adversarial_criterion(fake_output, real_label)

            g_loss = content_loss + adv_weight * adversarial_loss

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)
            optimizer_G.step()

            # Train Discriminator less frequently
            if batch_size % 2 == 0:
                optimizer_D.zero_grad()
                real_output = discriminator(hr_imgs)
                d_loss_real = adversarial_criterion(real_output, real_label)

                fake_output = discriminator(sr_imgs.detach())
                d_loss_fake = adversarial_criterion(fake_output, fake_label)

                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.5)
                optimizer_D.step()

            # Step schedulers
            scheduler_G.step()
            if batch_size % 2 == 0:
                scheduler_D.step()

            total_train_loss_G += g_loss.item()
            total_train_loss_D += d_loss.item()
            progress_bar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })

        avg_train_loss_G = total_train_loss_G / len(train_loader)
        avg_train_loss_D = total_train_loss_D / len(train_loader)

        # Evaluation
        val_psnr, val_ssim, val_ifc = evaluate_model(generator, val_loader, device, "Validation")
        set5_psnr, set5_ssim, set5_ifc = evaluate_model(generator, test_set5_loader, device, "Set5")
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

        # Save best model
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

        # Save checkpoint every 50 epochs
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'best_psnr': best_psnr,
            }, f'checkpoint_epoch_{epoch}.pth')

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
    plt.plot(epochs, history['val_psnr'], 'm-', label='Validation')
    plt.plot(epochs, history['set5_psnr'], 'g-', label='Set5')
    plt.title('PSNR Progress')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()

    # SSIM
    plt.subplot(1, 4, 3)
    plt.plot(epochs, history['val_ssim'], 'y-', label='Validation')
    plt.plot(epochs, history['set5_ssim'], 'g-', label='Set5')
    plt.title('SSIM Progress')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()

    # IFC
    plt.subplot(1, 4, 4)
    plt.plot(epochs, history['val_ifc'], 'c-', label='Validation')
    plt.plot(epochs, history['set5_ifc'], 'g-', label='Set5')
    plt.title('IFC Progress')
    plt.xlabel('Epoch')
    plt.ylabel('IFC')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves_4.png')
    plt.close()


def save_training_history(history, filename='training_history_4.txt'):
    with open(filename, 'w') as f:
        f.write(
            "Epoch,G_Loss,D_Loss,Val_PSNR,Val_SSIM,Val_IFC,Set5_PSNR,Set5_SSIM,Set5_IFC,Set14_PSNR,Set14_SSIM,Set14_IFC\n")

        for epoch in range(len(history['train_loss_G'])):
            f.write(f"{epoch + 1},"
                    f"{history['train_loss_G'][epoch]:.6f},"
                    f"{history['train_loss_D'][epoch]:.6f},"
                    f"{history['val_psnr'][epoch]:.4f},"
                    f"{history['val_ssim'][epoch]:.6f},"
                    f"{history['val_ifc'][epoch]:.6f},"
                    f"{history['set5_psnr'][epoch]:.4f},"
                    f"{history['set5_ssim'][epoch]:.6f},"
                    f"{history['set5_ifc'][epoch]:.6f},"
                    f"{history['set14_psnr'][epoch]:.4f},"
                    f"{history['set14_ssim'][epoch]:.6f},"
                    f"{history['set14_ifc'][epoch]:.6f}\n")


if __name__ == "__main__":
    # Configuration
    scale_factor = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    num_epochs = 2  # Increased from 300

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

    # Create datasets with augmentation
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

    try:
        print("\nStarting training with optimized parameters...")
        print("Phase 1 (Epochs 0-100): Content loss only")
        print("Phase 2 (Epochs 100-300): Content + small adversarial")
        print("Phase 3 (Epochs 300-500): Content + increased adversarial")

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
        print("Training curves plotted successfully")

        # Save history to file
        save_training_history(history)
        print("Training history saved successfully")

        # Final evaluation
        print("\nPerforming final evaluation...")

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
        print("\nFinal Results:", final_results)

        # Save final results to file
        with open('old/final_results_x4_3.txt', 'w') as f:
            f.write("Final Evaluation Results\n")
            f.write("=======================\n\n")
            for dataset, metrics in final_results.items():
                f.write(f"{dataset} Results:\n")
                f.write(f"PSNR: {metrics['PSNR']:.2f}\n")
                f.write(f"SSIM: {metrics['SSIM']:.4f}\n")
                f.write(f"IFC: {metrics['IFC']:.4f}\n\n")

        print("\nTraining and evaluation completed successfully!")

    except Exception as e:
        print(f"\nAn error occurred during training: {str(e)}")
        raise e