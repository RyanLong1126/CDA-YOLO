import torch
import torch.nn as nn

# ==================== PfAAM 模块 ====================
class PfAAM(nn.Module):
    """
    Parameter‑free Average Attention Module (PfAAM)
    Combines channel‑wise GAP and spatial‑wise average to generate 3D attention.
    No learnable parameters, only simple pooling and element‑wise operations.
    """
    def __init__(self, channels=None, out_channels=None):
        super(PfAAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @staticmethod
    def get_module_name():
        return "pfaam"

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel branch: GAP -> expand to full spatial size
        channel_att = self.avg_pool(x).view(b, c, 1, 1).expand_as(x)

        # Spatial branch: average over channels -> expand to same shape
        spatial_att = torch.mean(x, dim=1, keepdim=True).expand_as(x)

        # Combine and activate
        att = self.activation(channel_att * spatial_att)

        return x * att


# ==================== 基础组件（与原代码风格一致） ====================
def autopad(k, p=None, d=1):
    """Auto padding to keep output shape same as input."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with batch norm and activation."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k with customizable kernel size."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ==================== C2f 基础结构（使用 PfAAM 注意力） ====================
class C2f_PfAAM(nn.Module):
    """
    CSP Bottleneck with 2 convolutions + PfAAM attention.
    Similar to C2f_SimAM but uses PfAAM instead.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)                      # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
        self.att = PfAAM()                        # 使用 PfAAM 注意力

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))


# ==================== C3k2_PfAAM（用户需要的核心模块） ====================
class C3k2_PfAAM(C2f_PfAAM):
    """
    C3k2 variant with PfAAM attention.
    When c3k=True, uses C3k blocks; otherwise uses standard Bottleneck.
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        # Override the m list with either C3k or Bottleneck
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )