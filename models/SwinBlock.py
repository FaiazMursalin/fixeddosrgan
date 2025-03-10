from WindowAttention import *
from helpers.modelUtil import *


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution=None, num_heads=8, patch_size=2, depth_level=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        # Fixed window size
        self.window_size = 8
        self.shift_size = self.window_size // 2

        # Scale patch size with depth level - now allowing up to full window size
        # For example: depth_level 0, 1, 2, 3 could give patch sizes 2, 4, 6, 8
        self.patch_size = min(patch_size + (depth_level * 2), self.window_size)

        # Make number of heads dynamic based on patch size
        # Start with base_heads for smallest patch size, scale up with patch_size
        base_heads = 4  # Base number of heads for smallest patch size
        # Scale heads with patch_size ratio: more patch size = more heads
        self.num_heads = min(base_heads * (self.patch_size // patch_size), 16)

        # Ensure num_heads is at least 2 and divides dim evenly
        while self.num_heads > 2 and dim % self.num_heads != 0:
            self.num_heads -= 1

        self.depth_level = depth_level
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=self.num_heads,
            patch_size=self.patch_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        # print(f"Block with depth_level={depth_level}, patch_size={self.patch_size}, num_heads={self.num_heads}")

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Apply shifting only if window size > 1
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