import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils_2 import SRDataset, evaluate_model, plot_training_curves, save_results
from old.model8 import EnhancedGenerator, ContentLoss, TVLoss
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from tqdm import tqdm

def train_model(generator, train_loader, val_loader, test_set5_loader, test_set14_loader, num_epochs, device):
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    criterion_pixel = nn.L1Loss().to(device)
    criterion_content = ContentLoss().to(device)
    criterion_tv = TVLoss().to(device)
    
    optimizer_G = optim.AdamW(generator.parameters(), lr=8e-5, betas=(0.9, 0.999), weight_decay=1e-4)
    
    warmup_epochs = 5
    scheduler_G = optim.lr_scheduler.OneCycleLR(
        optimizer_G,
        max_lr=1e-4,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=warmup_epochs/num_epochs,
        anneal_strategy='cos'
    )
    
    scaler = GradScaler()
    content_weight = 0.15
    tv_weight = 0.00001

    history = {
        'train_loss_G': [], 'train_loss_pixel': [], 'train_loss_content': [], 'train_loss_tv': [],
        'val_psnr': [], 'val_ssim': [],
        'set5_psnr': [], 'set5_ssim': [],
        'set14_psnr': [], 'set14_ssim': [],
        'learning_rates': []
    }

    model_save_path = 'old/enhanced_model.pth'
    best_model_path = 'old/best_model.pth'
    best_psnr = 0

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ])

    for epoch in range(num_epochs):
        generator.train()
        total_g_loss = 0
        total_pixel_loss = 0
        total_content_loss = 0
        total_tv_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for lr_imgs, hr_imgs in progress_bar:
            if epoch > warmup_epochs:
                for i in range(lr_imgs.size(0)):
                    if torch.rand(1) < 0.7:  # Increased augmentation probability
                        lr_imgs[i] = train_transforms(lr_imgs[i])
                        hr_imgs[i] = train_transforms(hr_imgs[i])

            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer_G.zero_grad(set_to_none=True)  # Memory optimization

            with autocast():
                sr_imgs = generator(lr_imgs)
                
                pixel_loss = criterion_pixel(sr_imgs, hr_imgs)
                content_loss = criterion_content(sr_imgs, hr_imgs)
                tv_loss = criterion_tv(sr_imgs)
                
                g_loss = pixel_loss + content_weight * content_loss + tv_weight * tv_loss

            scaler.scale(g_loss).backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)  # Reduced for stability
            
            scaler.step(optimizer_G)
            scaler.update()
            scheduler_G.step()

            total_g_loss += g_loss.item()
            total_pixel_loss += pixel_loss.item()
            total_content_loss += content_loss.item()
            total_tv_loss += tv_loss.item()

            current_lr = optimizer_G.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss G': f'{g_loss.item():.4f}',
                'LR': f'{current_lr:.6f}'
            })

        avg_g_loss = total_g_loss / len(train_loader)
        avg_pixel_loss = total_pixel_loss / len(train_loader)
        avg_content_loss = total_content_loss / len(train_loader)
        avg_tv_loss = total_tv_loss / len(train_loader)

        generator.eval()
        with torch.no_grad():
            val_metrics = evaluate_model(generator, val_loader, device, "Validation")
            set5_metrics = evaluate_model(generator, test_set5_loader, device, "Set5")
            set14_metrics = evaluate_model(generator, test_set14_loader, device, "Set14")

        if set5_metrics[0] > best_psnr:
            best_psnr = set5_metrics[0]
            torch.save(generator.state_dict(), best_model_path)

        history['train_loss_G'].append(avg_g_loss)
        history['train_loss_pixel'].append(avg_pixel_loss)
        history['train_loss_content'].append(avg_content_loss)
        history['train_loss_tv'].append(avg_tv_loss)
        history['val_psnr'].append(val_metrics[0])
        history['val_ssim'].append(val_metrics[1])
        history['set5_psnr'].append(set5_metrics[0])
        history['set5_ssim'].append(set5_metrics[1])
        history['set14_psnr'].append(set14_metrics[0])
        history['set14_ssim'].append(set14_metrics[1])
        history['learning_rates'].append(current_lr)

        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_psnr': best_psnr,
            'history': history
        }, model_save_path)

        print(f'\nEpoch {epoch+1} Results:')
        print(f'Loss G: {avg_g_loss:.4f} (Pixel: {avg_pixel_loss:.4f}, Content: {avg_content_loss:.4f}, TV: {avg_tv_loss:.4f})')
        print(f'Set5 - PSNR: {set5_metrics[0]:.2f}, SSIM: {set5_metrics[1]:.4f}')
        print(f'Set14 - PSNR: {set14_metrics[0]:.2f}, SSIM: {set14_metrics[1]:.4f}')

    return history

if __name__ == "__main__":
    scale_factor = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 100

    print(f"Using device: {device}")
    generator = EnhancedGenerator(scale_factor=scale_factor).to(device)

    # Dataset paths remain the same
    general100_hr_dir = './dataset_improved/general100/x4/hr'
    general100_lr_dir = './dataset_improved/general100/x4/lr'
    set5_hr_dir = './dataset_improved/set5/x4/hr'
    set5_lr_dir = './dataset_improved/set5/x4/lr'
    set14_hr_dir = './dataset_improved/set14/x4/hr'
    set14_lr_dir = './dataset_improved/set14/x4/lr'

    general100_dataset = SRDataset(general100_hr_dir, general100_lr_dir)
    set5_dataset = SRDataset(set5_hr_dir, set5_lr_dir)
    set14_dataset = SRDataset(set14_hr_dir, set14_lr_dir)

    train_size = int(0.8 * len(general100_dataset))
    val_size = len(general100_dataset) - train_size
    train_dataset, val_dataset = random_split(
        general100_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_set5_loader = DataLoader(set5_dataset, batch_size=1)
    test_set14_loader = DataLoader(set14_dataset, batch_size=1)

    history = train_model(
        generator=generator,
        train_loader=train_loader,
        val_loader=val_loader,
        test_set5_loader=test_set5_loader,
        test_set14_loader=test_set14_loader,
        num_epochs=num_epochs,
        device=device
    )

    plot_training_curves(history)
    
    generator.eval()
    with torch.no_grad():
        set5_psnr, set5_ssim = evaluate_model(generator, test_set5_loader, device, "Set5")
        set14_psnr, set14_ssim = evaluate_model(generator, test_set14_loader, device, "Set14")

    final_metrics = {
        'Set5': {'psnr': set5_psnr, 'ssim': set5_ssim},
        'Set14': {'psnr': set14_psnr, 'ssim': set14_ssim}
    }

    save_results(history, final_metrics)
