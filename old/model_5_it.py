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

class ResidualBlock(nn.Module):
    def __init__(self, n_filters):
        super(ResidualBlock, self).__init__()
        expanded_filters = n_filters * 4  # Increased expansion ratio
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(n_filters, expanded_filters, kernel_size=1),  # Added bottleneck
            nn.BatchNorm2d(expanded_filters),
            nn.PReLU(),  # Replaced LeakyReLU with PReLU
            nn.Conv2d(expanded_filters, expanded_filters, kernel_size=3, padding=1, groups=expanded_filters),  # Depthwise
            nn.BatchNorm2d(expanded_filters),
            nn.PReLU(),
            nn.Conv2d(expanded_filters, n_filters, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(n_filters)
        )
        
        self.se = SELayer(n_filters)  # Added Squeeze-and-Excitation
        
    def forward(self, x):
        out = self.conv_block(x)
        out = self.se(out)
        return x + out  # Removed scaling factor for better gradient flow

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ProposedGenerator(nn.Module):
    def __init__(self, in_channels=3, n_filters=64, n_blocks=23, scale_factor=4):
        super(ProposedGenerator, self).__init__()
        
        # Enhanced initial feature extraction with gradient checkpointing
        self.initial_feature = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1),  # Smaller kernel
            nn.PReLU(),
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(n_filters * 2, n_filters, kernel_size=3, padding=1),
            nn.PReLU()
        )
        
        # Residual Groups with Local and Global Skip Connections
        self.residual_groups = nn.ModuleList([])
        n_groups = 4
        blocks_per_group = n_blocks // n_groups
        
        for _ in range(n_groups):
            group = []
            for _ in range(blocks_per_group):
                group.append(ResidualBlock(n_filters))
            self.residual_groups.append(nn.Sequential(*group))
            
        # Multi-scale feature fusion with 1x1 convs
        self.group_convs = nn.ModuleList([
            nn.Conv2d(n_filters, n_filters, kernel_size=1)
            for _ in range(n_groups)
        ])
        
        # Channel attention for global features
        self.global_attention = SELayer(n_filters)
        
        # Improved upsampling with sub-pixel convolution
        self.upsampling = nn.ModuleList()
        log_scale = int(torch.log2(torch.tensor(scale_factor)))
        
        for _ in range(log_scale):
            self.upsampling.extend([
                nn.Conv2d(n_filters, n_filters * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
                SELayer(n_filters)
            ])
        
        # Enhanced final reconstruction with residual learning
        self.final = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(n_filters, in_channels, kernel_size=3, padding=1)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        # Initial feature extraction
        initial_features = self.initial_feature(x)
        
        # Process through residual groups
        group_outputs = []
        features = initial_features
        
        for group, group_conv in zip(self.residual_groups, self.group_convs):
            group_out = group(features)
            group_outputs.append(group_conv(group_out))
            features = group_out
            
        # Multi-scale fusion with attention
        fused_features = sum(group_outputs)
        fused_features = self.global_attention(fused_features)
        
        # Global residual connection
        up_features = fused_features + initial_features
        
        # Progressive upsampling
        for up_block in self.upsampling:
            up_features = up_block(up_features)
            
        # Final reconstruction with residual connection
        out = self.final(up_features)
        return out + F.interpolate(x, scale_factor=4, mode='bicubic')

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.feature_extractor = FeatureExtractor()
        
    def forward(self, sr, hr):
        # Enhanced pixel loss
        pixel_loss = 0.9 * self.l1_loss(sr, hr) + 0.1 * self.mse_loss(sr, hr)
        
        # reduced Multi-level perceptual loss
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        perceptual_loss = (
            0.5 * self.mse_loss(sr_features, hr_features) +
            0.5 * self.l1_loss(sr_features, hr_features)
        )
        
        return pixel_loss + 0.01 * perceptual_loss

class ProposedDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(ProposedDiscriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, stride=1, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        # Enhanced feature extraction
        self.feature_extraction = nn.Sequential(
            discriminator_block(input_channels, 64, stride=2, normalize=False),
            discriminator_block(64, 128, stride=2),
            discriminator_block(128, 256, stride=2),
            discriminator_block(256, 512, stride=1),
            discriminator_block(512, 512, stride=1),
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        features = self.feature_extraction(x)
        validity = self.classifier(features)
        return validity.view(-1)
