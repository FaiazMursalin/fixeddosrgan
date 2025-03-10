from models.SwinBlock import *
from models.ChannelAttention import *


# Multi-scale window attention
class ProposedGenerator(nn.Module):
    def __init__(self, in_channels=3, n_filters=64, n_blocks=23, scale_factor=2, initial_patch_size=2):
        super(ProposedGenerator, self).__init__()

        self.initial_patch_size = initial_patch_size

        # Initial feature extraction
        self.initial_feature = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Swin Transformer blocks with Channel Attention
        # For window_size=8, initial_patch_size=2, we want to progress as 2,4,6,8
        # So we need 4 distinct depth levels (0,1,2,3)
        depth_increment = max(1, n_blocks // 4)  # At least get all 4 levels

        # Print expected progression info
        print(f"Patch size progression with window size 8:")
        for depth in range(4):
            patch_size = min(initial_patch_size + (depth * 2), 8)
            num_patches = (8 // patch_size) ** 2
            print(f"Depth level {depth}: patch_size={patch_size}×{patch_size}, " +
                  f"patches per window={num_patches}, " +
                  f"estimated attention heads={(4 * (patch_size // initial_patch_size))}")

        self.swin_blocks = nn.ModuleList([
            nn.Sequential(
                SwinBlock(
                    dim=n_filters,
                    input_resolution=None,
                    num_heads=8,  # Initial heads will be adjusted by SwinBlock based on patch size
                    patch_size=initial_patch_size,
                    depth_level=min(i // depth_increment, 3),  # Cap at depth level 3
                ),
                ChannelAttention(n_filters)
            ) for i in range(n_blocks)
        ])

        # Rest of the architecture remains the same
        self.fusion = nn.Sequential(
            nn.Conv2d(n_filters * n_blocks, n_filters, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        )

        self.global_skip_conv = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)

        self.upsampling = nn.ModuleList([])
        log_scale = int(torch.log2(torch.tensor(scale_factor)))
        for _ in range(log_scale):
            self.upsampling.extend([
                nn.Conv2d(n_filters, n_filters * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                ChannelAttention(n_filters)
            ])

        self.final = nn.Sequential(
            nn.Conv2d(n_filters, n_filters // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters // 2, in_channels, kernel_size=9, padding=4),
            nn.Tanh()
        )

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
        # Initial feature extraction
        initial_features = self.initial_feature(x)

        # Pad to multiple of fixed window size
        features, pad_h, pad_w = pad_to_multiple(initial_features, 8)

        # Process through Swin blocks with Channel Attention
        B, C, H, W = features.shape

        # Convert to sequence format for Swin blocks
        features = features.permute(0, 2, 3, 1).view(B, H * W, C)

        # Store intermediate features for multi-scale fusion
        block_outputs = []

        # Process through Swin blocks
        for block in self.swin_blocks:
            block[0].input_resolution = (H, W)
            features = block[0](features)
            # Convert back to spatial format for Channel Attention
            features_spatial = features.view(B, H, W, C).permute(0, 3, 1, 2)
            features_spatial = block[1](features_spatial)
            # Store the spatial features
            block_outputs.append(features_spatial)
            # Convert back to sequence format
            features = features_spatial.permute(0, 2, 3, 1).view(B, H * W, C)

        # Multi-scale feature fusion
        dense_features = torch.cat(block_outputs, dim=1)
        fused_features = self.fusion(dense_features)

        # Reshape back to spatial format
        features = features.view(B, H, W, C).permute(0, 3, 1, 2)

        # Remove padding if necessary
        if pad_h > 0 or pad_w > 0:
            features = features[:, :, :H - pad_h, :W - pad_w]

        # Global residual connection
        up_features = self.global_skip_conv(features) + initial_features

        # Upsampling
        for up_block in self.upsampling:
            up_features = up_block(up_features)

        return self.final(up_features)




