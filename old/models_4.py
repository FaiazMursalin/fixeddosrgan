import torch
import torch.nn as nn
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8):  # Reduced from 16 to 8
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Added max pooling
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel * 2, channel // reduction, 1, padding=0, bias=True),
            nn.PReLU(),  # Changed from ReLU to PReLU
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, n_filters, reduction=8):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, 3, 1, 1)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, 1, 1)
        self.ca = ChannelAttention(n_filters, reduction)
        self.residual_scale = 1.2  # Increased from 1.0

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.ca(out)
        out = out * self.residual_scale
        return res + out


class RG(nn.Module):
    def __init__(self, n_filters, n_rcab=6):  # Increased from 4 to 6
        super(RG, self).__init__()
        self.rcabs = nn.ModuleList([RCAB(n_filters) for _ in range(n_rcab)])
        self.conv = nn.Conv2d(n_filters, n_filters, 3, 1, 1)
        self.residual_scale = 1.1

    def forward(self, x):
        res = x
        out = x
        for rcab in self.rcabs:
            out = rcab(out)
        out = self.conv(out)
        return res + out * self.residual_scale


class ImprovedSRCNN(nn.Module):
    def __init__(self, scale_factor=2):
        super(ImprovedSRCNN, self).__init__()

        n_feats = 128
        n_groups = 6  # Increased from 4
        n_rcab = 6  # Increased from 4

        # Initial feature extraction with larger kernel
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, n_feats, kernel_size=9, padding=4),  # Increased kernel size
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.PReLU()
        )

        # Residual Groups with Channel Attention
        self.residual_groups = nn.ModuleList([
            RG(n_feats, n_rcab) for _ in range(n_groups)
        ])

        # Global feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(n_feats * n_groups, n_feats, 1),  # Concatenation fusion
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )

        # Progressive upsampling
        n_upsamples = int(np.log2(scale_factor))
        self.upsampling = nn.ModuleList()
        current_channels = n_feats
        for _ in range(n_upsamples):
            self.upsampling.append(nn.Sequential(
                nn.Conv2d(current_channels, current_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                RCAB(current_channels, reduction=8),
                nn.PReLU()
            ))

        # Final reconstruction with gradual reduction
        self.final = nn.Sequential(
            nn.Conv2d(n_feats, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Initial feature extraction
        shallow_features = self.initial_conv(x)

        # Deep feature extraction
        deep_features = shallow_features
        group_outputs = []

        for rg in self.residual_groups:
            deep_features = rg(deep_features)
            group_outputs.append(deep_features)

        # Multi-scale fusion
        deep_features = torch.cat(group_outputs, dim=1)
        deep_features = self.fusion(deep_features)

        # Global residual learning
        deep_features = deep_features + shallow_features

        # Upsampling
        for up_block in self.upsampling:
            deep_features = up_block(deep_features)

        return self.final(deep_features)


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

            # Dense layers implemented as convolutions
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Global average pooling followed by final classification
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

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
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.01)

        # Adjusted weights
        self.mse_weight = 0.6  # Higher weight for PSNR
        self.l1_weight = 0.2  # Some L1 for edges
        self.smooth_l1_weight = 0.2  # Added smooth L1 for stability

    def forward(self, sr, hr):
        return (self.mse_weight * self.mse_loss(sr, hr) +
                self.l1_weight * self.l1_loss(sr, hr) +
                self.smooth_l1_weight * self.smooth_l1(sr, hr))