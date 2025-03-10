import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        # Use first 35 layers for feature extraction
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return self.feature_extractor(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
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
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class MSRB(nn.Module):
    def __init__(self, n_filters):
        super(MSRB, self).__init__()
        self.conv1_3 = nn.Conv2d(n_filters, n_filters//2, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv2d(n_filters, n_filters//2, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.channel_attention = ChannelAttention(n_filters)
        self.spatial_attention = SpatialAttention()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out1_3 = self.conv1_3(x)
        out1_5 = self.conv1_5(x)
        multi_scale = torch.cat([out1_3, out1_5], dim=1)
        out = self.conv2(self.lrelu(multi_scale))
        out = self.channel_attention(out)
        out = self.spatial_attention(out)
        return x + out

class ImprovedERDB(nn.Module):
    def __init__(self, n_filters, growth_rate):
        super(ImprovedERDB, self).__init__()
        self.conv1 = nn.Conv2d(n_filters, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_filters + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n_filters + 2 * growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(n_filters + 3 * growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(n_filters + 4 * growth_rate, n_filters, kernel_size=1)
        self.channel_attention = ChannelAttention(n_filters)
        self.spatial_attention = SpatialAttention()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        f1 = self.lrelu(self.conv1(x))
        f2 = self.lrelu(self.conv2(torch.cat((x, f1), dim=1)))
        f3 = self.lrelu(self.conv3(torch.cat((x, f1, f2), dim=1)))
        f4 = self.lrelu(self.conv4(torch.cat((x, f1, f2, f3), dim=1)))
        f5 = self.conv5(torch.cat((x, f1, f2, f3, f4), dim=1))
        out = x + f5
        out = self.channel_attention(out)
        out = self.spatial_attention(out)
        return out

class ImprovedGenerator(nn.Module):
    def __init__(self, in_channels=3, n_filters=64, n_blocks=16, growth_rate=32, scale_factor=4):
        super(ImprovedGenerator, self).__init__()
        
        # Initial feature extraction with multi-scale processing
        self.conv1_3 = nn.Conv2d(in_channels, n_filters//2, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv2d(in_channels, n_filters//2, kernel_size=5, padding=2)
        self.initial_feature = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        
        # Multi-scale residual blocks
        self.msrb_blocks = nn.ModuleList([MSRB(n_filters) for _ in range(4)])
        

        # Improved ERDB blocks
        self.erdb_blocks = nn.ModuleList([ImprovedERDB(n_filters, growth_rate) for _ in range(n_blocks)])
        
        # Progressive upsampling
        self.upsampling = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_filters, n_filters * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                MSRB(n_filters)
            ) for _ in range(2)  # For 4x upscaling
        ])
        
        # Final reconstruction
        self.final_conv = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        # Multi-scale initial feature extraction
        feat1 = self.conv1_3(x)
        feat2 = self.conv1_5(x)
        initial_features = self.lrelu(self.initial_feature(torch.cat([feat1, feat2], dim=1)))
        
        # Multi-scale residual feature extraction
        msrb_out = initial_features
        for msrb in self.msrb_blocks:
            msrb_out = msrb(msrb_out)
        
        # Dense feature extraction with improved ERDB blocks
        features = msrb_out
        erdb_features = []
        for erdb in self.erdb_blocks:
            features = erdb(features)
            erdb_features.append(features)
        
        # Global feature fusion
        global_features = sum(erdb_features) / len(erdb_features)
        
        # Progressive upsampling with attention
        up_features = global_features + initial_features
        for up_block in self.upsampling:
            up_features = up_block(up_features)
        
        return self.final_conv(up_features)

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.feature_extractor = FeatureExtractor()
        
    def forward(self, sr, hr):
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        return self.l1_loss(sr_features, hr_features)

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
        
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return (h_tv + w_tv) / (batch_size * count_h * count_w)
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
