import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from timm.models.layers import to_2tuple, trunc_normal_


def pad_to_multiple(x, window_size):
    _, _, h, w = x.size()
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, pad_h, pad_w


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


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution=None, num_heads=8, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.layer_scale_1 = nn.Parameter(torch.ones(dim) * 1e-4)
        self.layer_scale_2 = nn.Parameter(torch.ones(dim) * 1e-4)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=None)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels // reduction, channels // reduction, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.fc(out).view(b, c, 1, 1)
        return x * out.expand_as(x)


# Multi-scale window attention
class ProposedGenerator(nn.Module):
    def __init__(self, in_channels=3, n_filters=64, n_blocks=23, scale_factor=2):
        super(ProposedGenerator, self).__init__()

        # Initial feature extract3*3
        # Follow SWInIR use one 3*3
        self.initial_feature = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Swin Transformer blocks with Channel Attention
        self.swin_blocks = nn.ModuleList([
            nn.Sequential(
                SwinBlock(
                    # patch_size =
                    dim=n_filters,
                    input_resolution=None,  # Will be set in forward pass
                    num_heads=8,
                    # window_size= dependent on the patch size, end up with 4*4 windows/ 2*2,
                    shift_size=0 if (i % 2 == 0) else 4
                ),
                ChannelAttention(n_filters)
            ) for i in range(n_blocks)
        ])

        # Multi-scale feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(n_filters * n_blocks, n_filters, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        )

        # Global skip connection conv
        self.global_skip_conv = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)

        # Progressive upsampling
        self.upsampling = nn.ModuleList([])
        log_scale = int(torch.log2(torch.tensor(scale_factor)))
        for _ in range(log_scale):
            self.upsampling.extend([
                nn.Conv2d(n_filters, n_filters * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                ChannelAttention(n_filters)
            ])

        # Final reconstruction
        self.final = nn.Sequential(
            nn.Conv2d(n_filters, n_filters // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters // 2, in_channels, kernel_size=9, padding=4),
            nn.Tanh()
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
        # Initial feature extraction
        initial_features = self.initial_feature(x)

        # Pad to multiple of window size
        features, pad_h, pad_w = pad_to_multiple(initial_features, 8)  # 8 is the window size

        # Process through Swin blocks with Channel Attention
        B, C, H, W = features.shape

        # Convert to sequence format for Swin blocks
        features = features.permute(0, 2, 3, 1).view(B, H * W, C)

        # Store intermediate features for multi-scale fusion
        block_outputs = []

        # Update input resolution for each Swin block
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
        dense_features = torch.cat(block_outputs, dim=1)  # Concatenate along channel dimension
        fused_features = self.fusion(dense_features)

        # Reshape back to spatial format
        features = features.view(B, H, W, C).permute(0, 3, 1, 2)

        # Combine fused features with the final features
        # features = features + fused_features

        # Remove padding if necessary
        if pad_h > 0 or pad_w > 0:
            features = features[:, :, :H - pad_h, :W - pad_w]

        # Global residual connection
        up_features = self.global_skip_conv(features) + initial_features

        # Upsampling
        for up_block in self.upsampling:
            up_features = up_block(up_features)

        return self.final(up_features)


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