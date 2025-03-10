import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)

class ERDB(nn.Module):
    def __init__(self, n_filters, growth_rate):
        super(ERDB, self).__init__()
        self.conv1 = nn.Conv2d(n_filters, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_filters + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n_filters + 2 * growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(n_filters + 3 * growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(n_filters + 4 * growth_rate, n_filters, kernel_size=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        f1 = self.lrelu(self.conv1(x))
        f2 = self.lrelu(self.conv2(torch.cat((x, f1), dim=1)))
        f3 = self.lrelu(self.conv3(torch.cat((x, f1, f2), dim=1)))
        f4 = self.lrelu(self.conv4(torch.cat((x, f1, f2, f3), dim=1)))
        f5 = self.conv5(torch.cat((x, f1, f2, f3, f4), dim=1))
        return x + f5

class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels*2, channels // 8, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        attention = self.attention(torch.cat((x1, x2), dim=1))
        return attention * x1 + (1 - attention) * x2

class ProposedGenerator(nn.Module):
    def __init__(self, in_channels=3, n_filters=64, n_blocks=16, growth_rate=32, scale_factor=4):
        super(ProposedGenerator, self).__init__()
        
        # Initial feature extraction
        self.initial_feature = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # ERDBs with attention fusion
        self.erdb_blocks = nn.ModuleList([ERDB(n_filters, growth_rate) for _ in range(n_blocks)])
        self.attention_fusion = AttentionFusion(n_filters)
        
        # Global residual learning
        self.global_conv = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        
        # Upsampling
        self.upsampling = nn.Sequential(
            nn.Conv2d(n_filters, n_filters * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters, n_filters * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final reconstruction
        self.final_conv = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters, in_channels, kernel_size=9, padding=4),
            nn.Tanh()
        )
        
    def forward(self, x):
        initial_features = self.initial_feature(x)
        
        features = initial_features
        for erdb in self.erdb_blocks:
            erdb_features = erdb(features)
            features = self.attention_fusion(features, erdb_features)
        
        global_features = self.global_conv(features)
        upsampled_features = self.upsampling(global_features + initial_features)
        
        return self.final_conv(upsampled_features)

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.feature_extractor = FeatureExtractor()
        
    def forward(self, sr, hr):
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        
        return self.l1_loss(sr_features, hr_features)
