import torch
import torch.nn as nn


# ==================== ELA 模块 ====================
class ELA(nn.Module):
    """
    Efficient Local Attention (ELA)
    使用 1D 深度可分离卷积沿着 H 和 W 方向提取局部空间注意力。
    包含少量参数，具有强局部空间建模能力。
    """

    def __init__(self, channels, groups=4, kernel_size=7):
        super(ELA, self).__init__()
        # 确保 GroupNorm 的 groups 数能被通道数整除，否则回退为 1
        if channels % groups != 0:
            groups = 1

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 沿 H 方向的 1D 深度可分离卷积 + GroupNorm + Sigmoid
        self.conv_h = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size, 1), 1, (kernel_size // 2, 0), groups=channels),
            nn.GroupNorm(groups, channels),
            nn.Sigmoid()
        )

        # 沿 W 方向的 1D 深度可分离卷积 + GroupNorm + Sigmoid
        self.conv_w = nn.Sequential(
            nn.Conv2d(channels, channels, (1, kernel_size), 1, (0, kernel_size // 2), groups=channels),
            nn.GroupNorm(groups, channels),
            nn.Sigmoid()
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(channels={self.conv_h[0].in_channels})"

    @staticmethod
    def get_module_name():
        return "ela"

    def forward(self, x):
        # H 方向注意力分支
        x_h = self.pool_h(x)
        x_h = self.conv_h(x_h).expand_as(x)

        # W 方向注意力分支
        x_w = self.pool_w(x)
        x_w = self.conv_w(x_w).expand_as(x)

        # 融合注意力权重
        return x * x_h * x_w


# ==================== 基础组件（沿用你的代码） ====================
def autopad(k, p=None, d=1):
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


class C3(nn.Module):
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
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ==================== C2f 基础结构（使用 ELA 注意力） ====================
class C2f_ELA(nn.Module):
    """
    CSP Bottleneck with 2 convolutions + ELA attention.
    注意：ELA 是有参注意力，必须传入输出通道数 c2
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
        self.att = ELA(c2)  # 使用 ELA 注意力，需传入 c2

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))


# ==================== C3k2_ELA（核心模块） ====================
class C3k2_ELA(C2f_ELA):
    """
    C3k2 variant with ELA attention.
    When c3k=True, uses C3k blocks; otherwise uses standard Bottleneck.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        # Override the m list with either C3k or Bottleneck
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
