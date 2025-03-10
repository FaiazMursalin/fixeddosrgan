import torch
import torch.nn as nn
import numpy as np

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    """Residual Channel Attention Block"""
    def __init__(self, n_filters, reduction=16):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            ChannelAttention(n_filters, reduction)
        )
        self.residual_scale = 1.0

    def forward(self, x):
        res = self.body(x)
        res = res * self.residual_scale
        return res + x

class RG(nn.Module):
    """Residual Group"""
    def __init__(self, n_filters, n_rcab=4):
        super(RG, self).__init__()
        self.body = nn.Sequential(*[RCAB(n_filters) for _ in range(n_rcab)])
        self.conv = nn.Conv2d(n_filters, n_filters, 3, 1, 1)

    def forward(self, x):
        res = self.body(x)
        res = self.conv(res)
        return res + x

class ImprovedSRCNN(nn.Module):
    def __init__(self, scale_factor=2):
        super(ImprovedSRCNN, self).__init__()

        n_feats = 128  # Base number of filters
        n_groups = 4   # Number of residual groups
        n_rcab = 4    # Number of RCABs per group

        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, n_feats, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.PReLU()
        )

        # Residual Groups with Channel Attention
        self.residual_groups = nn.ModuleList([
            RG(n_feats, n_rcab) for _ in range(n_groups)
        ])
        
        # Global residual learning
        self.conv_last = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

        # Progressive upsampling
        n_upsamples = int(np.log2(scale_factor))
        self.upsampling = nn.ModuleList()
        current_channels = n_feats
        for _ in range(n_upsamples):
            self.upsampling.append(nn.Sequential(
                nn.Conv2d(current_channels, current_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                RCAB(current_channels),
                nn.PReLU()
            ))

        # Final reconstruction
        self.final = nn.Sequential(
            nn.Conv2d(n_feats, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Initial feature extraction
        x = self.initial_conv(x)
        residual = x

        # Deep feature extraction with residual groups
        for rg in self.residual_groups:
            x = rg(x)
        
        # Global residual learning
        x = self.conv_last(x)
        x = x + residual

        # Upsampling
        for up_block in self.upsampling:
            x = up_block(x)

        # Final reconstruction
        return self.final(x)

class Discriminator(nn.Module):
    def __init__(self, input_shape=(3, 96, 96)):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, normalize=True, kernel_size=4, stride=2, padding=1):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # Initial layer - no normalization
            *discriminator_block(3, 64, normalize=False, kernel_size=3, stride=1, padding=1),
            
            # Increase channel depth gradually
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            
            # Additional convolution layers with stride 1
            *discriminator_block(512, 512, kernel_size=3, stride=1, padding=1),
            *discriminator_block(512, 512, kernel_size=3, stride=1, padding=1),
            
            # Dense layers implemented as convolutions
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Global average pooling followed by final classification
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        return self.model(x).view(-1)

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Balanced weights for both PSNR and perceptual quality
        self.mse_weight = 0.75  # Slightly higher for PSNR
        self.l1_weight = 0.25   # Lower but still significant for edges

    def forward(self, sr, hr):
        return self.l1_weight * self.l1_loss(sr, hr) + self.mse_weight * self.mse_loss(sr, hr)
