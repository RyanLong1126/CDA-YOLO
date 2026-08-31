import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Helper Functions & Basic Blocks from Template ---

def autopad(k, p=None, d=1):
    """Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# --- Core Implementation: Monte Carlo Attention ---

class MCAttn(nn.Module):
    """
    Monte Carlo Attention (MCAttn) Module.
    Reference: "Exploiting Scale-Variant Attention for Segmenting Small Medical Objects" (SvANet)

    Uses random sampling (Monte Carlo) to select/weight attention maps from multiple scales (1x1, 2x2, 3x3)
    to preserve details of small objects.
    """

    def __init__(self, channels, scales=(1, 2, 3)):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        # Learnable logits for association probabilities P(x, i)
        self.logits = nn.Parameter(torch.zeros(self.num_scales))

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. Generate multi-scale feature maps (Pooling + Upsampling)
        pooled_features = []
        for s in self.scales:
            # Avoid pool size larger than feature map size
            pool_size = min(s, h, w)
            if pool_size == 0:
                continue
            # Average Pooling
            p = F.adaptive_avg_pool2d(x, pool_size)
            # Upsample back to original resolution for spatial attention
            up = F.interpolate(p, size=(h, w), mode='bilinear', align_corners=False)
            pooled_features.append(up)

        if not pooled_features:
            return x  # Fallback for very small feature maps

        # Stack along a new dimension: shape (B, num_scales, C, H, W)
        stacked = torch.stack(pooled_features, dim=1)

        # 2. Calculate Association Probabilities (Monte Carlo Sampling)
        if self.training:
            # Training: Use Gumbel-Softmax to simulate "Random Selection" (Monte Carlo process)
            # tau=1.0 controls the sharpness; hard=False allows gradient flow (soft selection approximation)
            # To strictly follow "random selection", you could set hard=True, but soft is often more stable for YOLO.
            probs = F.gumbel_softmax(self.logits.unsqueeze(0).expand(b, -1), tau=1.0, hard=False)
        else:
            # Inference: Use deterministic expected value (Weighted Sum)
            probs = F.softmax(self.logits.unsqueeze(0).expand(b, -1), dim=-1)

        # Reshape for broadcasting: (B, num_scales, 1, 1, 1)
        probs = probs.view(b, self.num_scales, 1, 1, 1)

        # 3. Aggregate features: Weighted sum based on probabilities
        attn_map = (stacked * probs).sum(dim=1)

        # 4. Generate Spatial Attention Map
        attn_map = torch.sigmoid(attn_map)

        return x * attn_map


# --- Integration: C3k2_MCAttn ---

class C2f_MCAttn(nn.Module):
    """CSP Bottleneck with 2 convolutions and MCAttn at the end."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = MCAttn(c2)  # Apply MCAttn

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))


class C3k2_MCAttn(C2f_MCAttn):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, C3k blocks, and MCAttn."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        # Replace Bottleneck with C3k if flag is True
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

