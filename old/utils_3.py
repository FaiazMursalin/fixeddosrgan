import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def calculate_metrics(img1, img2):
    # PSNR
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)
    
    # SSIM
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    window_size = 11
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
    
    return psnr.item(), ssim.mean().item()

def evaluate_model(model, test_loader, device, dataset_name):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    num_images = 0

    print(f"\nEvaluating on {dataset_name}...")
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(test_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            sr_imgs = model(lr_imgs)

            for i in range(sr_imgs.size(0)):
                psnr, ssim = calculate_metrics(sr_imgs[i], hr_imgs[i])
                total_psnr += psnr
                total_ssim += ssim
                num_images += 1

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images

    print(f"\nResults on {dataset_name}:")
    print(f"PSNR: {avg_psnr:.2f}")
    print(f"SSIM: {avg_ssim:.4f}")

    return avg_psnr, avg_ssim

def plot_training_curves(history, save_path='training_curves_optimized_esrgan.png'):
    epochs = range(1, len(history['train_loss_G']) + 1)

    plt.figure(figsize=(15, 5))

    # Generator and Discriminator Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss_G'], 'b-', label='Generator Loss')
    plt.plot(epochs, history['train_loss_D'], 'r-', label='Discriminator Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # PSNR
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['val_psnr'], 'g-', label='Validation')
    plt.plot(epochs, history['set5_psnr'], 'y-', label='Set5')
    plt.plot(epochs, history['set14_psnr'], 'm-', label='Set14')
    plt.title('PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()

    # SSIM
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_ssim'], 'g-', label='Validation')
    plt.plot(epochs, history['set5_ssim'], 'y-', label='Set5')
    plt.plot(epochs, history['set14_ssim'], 'm-', label='Set14')
    plt.title('SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_results(history, metrics, save_path='model_7.txt'):
    with open(save_path, 'w') as f:
        f.write("Training History\n")
        f.write("===============\n\n")
        for epoch in range(len(history['train_loss_G'])):
            f.write(f"Epoch {epoch + 1}:\n")
            f.write(f"G_Loss: {history['train_loss_G'][epoch]:.6f}\n")
            f.write(f"D_Loss: {history['train_loss_D'][epoch]:.6f}\n")
            f.write(f"Val_PSNR: {history['val_psnr'][epoch]:.4f}\n")
            f.write(f"Val_SSIM: {history['val_ssim'][epoch]:.6f}\n\n")

        f.write("\nFinal Metrics\n")
        f.write("=============\n\n")
        for dataset, values in metrics.items():
            f.write(f"{dataset}:\n")
            f.write(f"PSNR: {values['psnr']:.4f}\n")
            f.write(f"SSIM: {values['ssim']:.6f}\n\n")
