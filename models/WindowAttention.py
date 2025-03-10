import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
import math


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads=8, patch_size=None, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if patch_size is None:
            self.patch_size = max(window_size[0] // 2, 1)
        else:
            # Allow patch size to be up to full window size
            self.patch_size = min(patch_size, window_size[0])

        # Calculate number of patches, handling the edge case where patch_size = window_size
        self.num_patches_h = max(window_size[0] // self.patch_size, 1)
        self.num_patches_w = max(window_size[1] // self.patch_size, 1)
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Create appropriate patch embedding based on patch size
        if self.patch_size >= window_size[0]:
            # Special case for single patch (entire window)
            self.patch_embed = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(dim, dim, kernel_size=1)  # Preserve dimensions
            )
        else:
            # Regular patching for multiple patches
            self.patch_embed = nn.Conv2d(
                dim, dim,
                kernel_size=self.patch_size,
                stride=self.patch_size
            )

        # Layer norm that operates on the final dimension only
        self.norm = nn.LayerNorm(dim)

        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, self.num_patches, self.num_patches))
        trunc_normal_(self.rel_pos_bias, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        h = w = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B_, C, h, w)

        # Apply patching
        x = self.patch_embed(x)  # B_, C, patches_h, patches_w

        # Reshape for normalization and attention
        if self.num_patches == 1:
            # Single patch case
            x = x.view(B_, C, 1, 1)
            x = x.flatten(2)  # B_, C, 1
            x = x.transpose(1, 2)  # B_, 1, C
        else:
            # Multiple patches case
            x = x.flatten(2)  # B_, C, num_patches
            x = x.transpose(1, 2)  # B_, num_patches, C

        # Apply normalization
        x = self.norm(x)

        # Compute QKV - safer approach without reshape
        qkv = self.qkv(x)  # B_, num_patches, 3*C

        # Split QKV directly without reshape
        chunk_size = C
        q = qkv[:, :, :chunk_size]
        k = qkv[:, :, chunk_size:2 * chunk_size]
        v = qkv[:, :, 2 * chunk_size:]

        # Reshape for multi-head attention - do this safely by calculating proper sizes
        head_dim = C // self.num_heads

        # Safe reshape for q, k, v
        q = q.reshape(B_, self.num_patches, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B_, self.num_patches, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B_, self.num_patches, self.num_heads, head_dim).permute(0, 2, 1, 3)

        # Apply attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        attn = attn + self.rel_pos_bias.unsqueeze(0)

        # Apply mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, self.num_patches, self.num_patches) + mask.unsqueeze(
                1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, self.num_patches, self.num_patches)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Apply attention weights
        x = (attn @ v).transpose(1, 2).reshape(B_, self.num_patches, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # For the case of a single patch, special upsampling is needed
        if self.num_patches == 1:
            # Reshape to [B_, 1, 1, C] then upsample to [B_, h, w, C]
            x = x.view(B_, 1, 1, C).expand(B_, h, w, C)
        else:
            # Reshape back to 2D spatial layout
            x = x.view(B_, self.num_patches_h, self.num_patches_w, C)
            # Upsample back to original window size
            x = x.permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(h, w), mode='bicubic', align_corners=False)
            x = x.permute(0, 2, 3, 1)

        # Reshape back to sequence format
        x = x.reshape(B_, N, C)

        return x