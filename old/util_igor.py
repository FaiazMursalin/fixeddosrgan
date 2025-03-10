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
from datetime import datetime

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

def plot_training_curves(history, save_path='igor_training_curves.png'):
    """
    Plot training curves using only matplotlib
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure
    plt.figure(figsize=(20, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Total Loss', linewidth=2)
    if 'content_loss' in history:
        plt.plot(epochs, history['content_loss'], 'g--', label='Content Loss')
    if 'perceptual_loss' in history:
        plt.plot(epochs, history['perceptual_loss'], 'r--', label='Perceptual Loss')
    if 'adversarial_loss' in history:
        plt.plot(epochs, history['adversarial_loss'], 'm--', label='Adversarial Loss')
    plt.title('Training Losses', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # PSNR plot
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['val_psnr'], 'g-', label='Validation', linewidth=2)
    plt.plot(epochs, history['set5_psnr'], 'y-', label='Set5', linewidth=2)
    plt.plot(epochs, history['set14_psnr'], 'm-', label='Set14', linewidth=2)
    if 'urban100_psnr' in history:
        plt.plot(epochs, history['urban100_psnr'], 'c-', label='Urban100', linewidth=2)
    plt.title('PSNR Metrics', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # SSIM plot
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['val_ssim'], 'g-', label='Validation', linewidth=2)
    plt.plot(epochs, history['set5_ssim'], 'y-', label='Set5', linewidth=2)
    plt.plot(epochs, history['set14_ssim'], 'm-', label='Set14', linewidth=2)
    if 'urban100_ssim' in history:
        plt.plot(epochs, history['urban100_ssim'], 'c-', label='Urban100', linewidth=2)
    plt.title('SSIM Metrics', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Learning rate plot
    if 'learning_rate' in history:
        plt.subplot(2, 2, 4)
        plt.semilogy(epochs, history['learning_rate'], 'b-', linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_results(history, metrics, save_path='igor_training_results.txt'):
    """
    Save training results to a text file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(save_path, 'w') as f:
        # Header
        f.write(f"Training Results - {timestamp}\n")
        f.write("="* 50 + "\n\n")
        
        # Training History
        f.write("Training History\n")
        f.write("---------------\n")
        for epoch in range(len(history['train_loss'])):
            f.write(f"\nEpoch {epoch + 1}:\n")
            f.write(f"Loss: {history['train_loss'][epoch]:.6f}\n")
            
            if 'content_loss' in history:
                f.write(f"Content Loss: {history['content_loss'][epoch]:.6f}\n")
            if 'perceptual_loss' in history:
                f.write(f"Perceptual Loss: {history['perceptual_loss'][epoch]:.6f}\n")
            if 'adversarial_loss' in history:
                f.write(f"Adversarial Loss: {history['adversarial_loss'][epoch]:.6f}\n")
            
            # Metrics
            f.write("\nMetrics:\n")
            f.write(f"Validation PSNR: {history['val_psnr'][epoch]:.4f} dB\n")
            f.write(f"Validation SSIM: {history['val_ssim'][epoch]:.6f}\n")
            f.write(f"Set5 PSNR: {history['set5_psnr'][epoch]:.4f} dB\n")
            f.write(f"Set5 SSIM: {history['set5_ssim'][epoch]:.6f}\n")
            f.write(f"Set14 PSNR: {history['set14_psnr'][epoch]:.4f} dB\n")
            f.write(f"Set14 SSIM: {history['set14_ssim'][epoch]:.6f}\n")
            
            if 'urban100_psnr' in history:
                f.write(f"Urban100 PSNR: {history['urban100_psnr'][epoch]:.4f} dB\n")
                f.write(f"Urban100 SSIM: {history['urban100_ssim'][epoch]:.6f}\n")
        
        # Best Performance
        f.write("\nBest Performance\n")
        f.write("---------------\n")
        best_epoch = np.argmax(history['val_psnr'])
        f.write(f"Best Validation PSNR at epoch {best_epoch + 1}:\n")
        f.write(f"  PSNR: {history['val_psnr'][best_epoch]:.4f} dB\n")
        f.write(f"  SSIM: {history['val_ssim'][best_epoch]:.6f}\n")
        
        # Final Metrics
        f.write("\nFinal Metrics\n")
        f.write("-------------\n")
        for dataset, values in metrics.items():
            f.write(f"\n{dataset.upper()}:\n")
            if isinstance(values, dict):
                for metric, value in values.items():
                    f.write(f"  {metric}: {value:.6f}\n")
