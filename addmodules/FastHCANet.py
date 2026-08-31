import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
import os

sys.path.append(os.getcwd())


# m_seed = 1
# # 设置seed
# torch.manual_seed(m_seed)
# torch.cuda.manual_seed_all(m_seed)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Multi-Scale Feed-Forward Network (MSFN)
class MSFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        # 全部改为 2D 卷积
        self.project_in = nn.Conv2d(dim, hidden_features * 3, kernel_size=1, bias=bias)


        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1,
                                 stride=1, dilation=1, padding=0,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3,
                                 stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5,
                                 stride=1, padding=2,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.project_in(x)  # (B, 3*hidden, H, W)
        x1, x2, x3 = x.chunk(3, dim=1)  # 各 (B, hidden, H, W)

        x1 = self.dwconv1(x1)  # dilation=1
        x2 = self.dwconv2(x2)  # dilation=2
        x3 = self.dwconv3(x3)  # dilation=3

        x = F.gelu(x1) * x2 * x3  # 逐元素乘

        x = self.project_out(x)  # (B, dim, H, W)
        return x


##########################################################################
## Convolution and Attention Fusion Module  (CAFM)
class CAFMAttention_Fast(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5  # 固定缩放，替代 temperature + L2 norm

        # 纯 2D，无 3D 包装
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1,
                                    groups=dim * 3, bias=bias)

        # 静态深度可分离卷积，替代动态权重生成
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
        )

        # 拼接后投影，替代直接相加
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.pos_embed = nn.Parameter(torch.randn(1, dim, 256, 256) * 0.02)
    def forward(self, x):
        b, c, h, w = x.shape
        pos = F.interpolate(self.pos_embed, size=(h, w), mode='bilinear', align_corners=False)
        x = x + pos  # 直接在原始特征上加位置信息

        # 纯 2D，无 squeeze/unsqueeze
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 全局注意力（标准实现，无 L2 norm）
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # 固定缩放
        attn = attn.softmax(dim=-1)
        out_global = attn @ v
        out_global = rearrange(out_global, 'b head c (h w) -> b (head c) h w',
                               head=self.num_heads, h=h, w=w)

        # 局部卷积（静态，无需动态生成权重）
        out_local = self.local_conv(x)

        # 拼接融合，替代直接相加（让网络自己学习融合权重）
        out = torch.cat([out_global, out_local], dim=1)
        out = self.project_out(out)
        return out




##########################################################################
## CAMixing Block
class FastCAMixingTransformer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(FastCAMixingTransformer, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CAFMAttention_Fast(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = MSFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x