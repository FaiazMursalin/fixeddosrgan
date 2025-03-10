import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return self.feature_extractor(x)

class RCAB(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            ChannelAttention(channels, reduction)
        )
    
    def forward(self, x):
        return x + self.body(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out

class EnhancedMSRB(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.conv1_3 = nn.Conv2d(n_filters, n_filters//2, 3, padding=1)
        self.conv1_5 = nn.Conv2d(n_filters, n_filters//2, 3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.rcab = RCAB(n_filters)
        self.spatial_attention = SpatialAttention()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out1_3 = self.conv1_3(x)
        out1_5 = self.conv1_5(x)
        multi_scale = torch.cat([out1_3, out1_5], dim=1)
        out = self.conv2(self.lrelu(multi_scale))
        out = self.rcab(out)
        out = self.spatial_attention(out)
        return x + out

class DenseResidualBlock(nn.Module):
    def __init__(self, n_filters, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_filters + i * growth_rate, growth_rate, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ) for i in range(4)
        ])
        self.conv_fusion = nn.Conv2d(n_filters + 4 * growth_rate, n_filters, 1)
        self.rcab = RCAB(n_filters)
        
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        out = self.conv_fusion(torch.cat(features, dim=1))
        out = self.rcab(out)
        return x + out

class EnhancedGenerator(nn.Module):
    def __init__(self, in_channels=3, n_filters=64, n_blocks=16, growth_rate=32, scale_factor=4):
        super().__init__()
        
        # Initial feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, n_filters//2, 3, padding=1),
            RCAB(n_filters//2),
            nn.Conv2d(n_filters//2, n_filters, 3, padding=1)
        )
        
        # Multi-scale feature extraction
        self.msrb_blocks = nn.ModuleList([EnhancedMSRB(n_filters) for _ in range(4)])
        
        # Dense feature extraction
        self.dense_blocks = nn.ModuleList([DenseResidualBlock(n_filters, growth_rate) for _ in range(n_blocks)])
        
        # Progressive upsampling
        self.upsampling = nn.ModuleList([
            nn.Sequential(
                RCAB(n_filters),
                nn.Conv2d(n_filters, n_filters * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                EnhancedMSRB(n_filters)
            ) for _ in range(2)
        ])
        
        # Final reconstruction
        self.final = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters, in_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Initial features
        init_features = self.conv1(x)
        
        # Multi-scale features
        msrb_out = init_features
        for msrb in self.msrb_blocks:
            msrb_out = msrb(msrb_out)
        
        # Dense features
        dense_out = msrb_out
        dense_features = []
        for block in self.dense_blocks:
            dense_out = block(dense_out)
            dense_features.append(dense_out)
            
        # Global feature fusion
        global_features = sum(dense_features) / len(dense_features)
        
        # Upsampling
        up_features = global_features + init_features
        for up_block in self.upsampling:
            up_features = up_block(up_features)
            
        return self.final(up_features)

class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.feature_extractor = FeatureExtractor()
        
    def forward(self, sr, hr):
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        return self.l1_loss(sr_features, hr_features)

class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv = torch.pow((x[:,:,1:,:] - x[:,:,:h_x-1,:]), 2).sum()
        w_tv = torch.pow((x[:,:,:,1:] - x[:,:,:,:w_x-1]), 2).sum()
        return (h_tv + w_tv) / (batch_size * x.size()[1] * h_x * w_x)
