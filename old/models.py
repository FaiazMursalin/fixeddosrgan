import torch
import torch.nn as nn
import numpy as np


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):  # Changed reduction from 8 to 16
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y = torch.cat([y_avg, y_max], dim=1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EnhancedResidualBlock(nn.Module):
    def __init__(self, n_filters):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters * 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters * 4)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(n_filters * 4, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.se = SELayer(n_filters)
        self.residual_scale = 0.2

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = out * self.residual_scale + identity
        out = self.prelu(out)
        return out


class ImprovedSRCNN(nn.Module):
    def __init__(self, scale_factor=2):
        super(ImprovedSRCNN, self).__init__()

        # Enhanced initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.PReLU()
        )

        # More residual blocks for better feature extraction
        self.residual_layers = nn.ModuleList([
            EnhancedResidualBlock(256) for _ in range(16)
        ])

        # Enhanced fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            SELayer(256, reduction=16),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.PReLU()
        )

        # Improved upsampling blocks
        n_upsamples = int(np.log2(scale_factor))
        self.upsampling = nn.ModuleList()
        for _ in range(n_upsamples):
            self.upsampling.append(nn.Sequential(
                nn.Conv2d(256, 1024, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                SELayer(256),
                nn.PReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.PReLU()
            ))

        # Enhanced reconstruction layer
        self.final = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Initial feature extraction
        x = self.initial_conv(x)
        residual = x

        # Deep feature extraction with residual learning
        features = []
        for res_block in self.residual_layers:
            x = res_block(x)
            features.append(x)

        # Enhanced feature fusion
        x = torch.cat(features[-6:], dim=1)  # Use last 6 feature maps
        x = nn.Conv2d(256 * 6, 256, kernel_size=1).to(x.device)(x)
        x = self.fusion(x)
        x = x + residual

        # Progressive upsampling
        for up_block in self.upsampling:
            x = up_block(x)

        # Final reconstruction
        return self.final(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, stride=1, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64, 2, normalize=False),
            *discriminator_block(64, 128, 2),
            *discriminator_block(128, 256, 2),
            *discriminator_block(256, 512, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1)


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        # Adjusted weights to favor MSE for better PSNR
        self.mse_weight = 0.7
        self.l1_weight = 0.3

    def forward(self, sr, hr):
        return self.l1_weight * self.l1_loss(sr, hr) + self.mse_weight * self.mse_loss(sr, hr)