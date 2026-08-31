import torch
import torch.nn as nn

# ==================== PfAAMR 模块（改进版） ====================
class PfAAMR(nn.Module):
    """
    Parameter‑free Average Attention Module (PfAAM) – 改进版
    - 利用广播机制，避免 expand_as 操作，更高效
    - 可选残差连接：x * (1 + att) 代替 x * att（默认关闭，保持原功能）
    """
    def __init__(self, channels=None, out_channels=None, residual=False):
        super(PfAAMR, self).__init__()
        self.activation = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.residual = residual   # 创新点：可选残差形式，默认 False = 原功能

    def __repr__(self):
        return f"{self.__class__.__name__}(residual={self.residual})"

    @staticmethod
    def get_module_name():
        return "pfaamr"

    def forward(self, x):
        b, c, h, w = x.shape

        # 通道分支：GAP → (b, c, 1, 1)
        channel_att = self.avg_pool(x)          # (b, c, 1, 1)
        # 空间分支：通道均值 → (b, 1, h, w)
        spatial_att = torch.mean(x, dim=1, keepdim=True)  # (b, 1, h, w)

        # 广播相乘得到 (b, c, h, w)，避免 expand_as
        att = self.activation(channel_att * spatial_att)

        if self.residual:
            return x * (1 + att)   # 轻量残差连接，梯度更平滑
        else:
            return x * att         # 原功能（默认）


# ==================== 基础组件（与原代码一致） ====================
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


# ==================== C2f 基础结构（整合改进版 PfAAM） ====================
class C2f_PfAAMR(nn.Module):
    """
    CSP Bottleneck with 2 convolutions + PfAAM attention.
    支持传递 residual 参数给 PfAAM。
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, att_residual=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
        self.att = PfAAMR(residual=att_residual)   # 允许开启残差注意力

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))


# ==================== C3k2_PfAAM（用户所需核心模块） ====================
class C3k2_PfAAMR(C2f_PfAAMR):
    """
    C3k2 variant with PfAAMR attention.
    支持 c3k 标志切换 C3k 或 Bottleneck，同时支持 att_residual。
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, att_residual=True):
        # 先调用父类初始化（暂时构建默认的 Bottleneck 列表）
        super().__init__(c1, c2, n, shortcut, g, e, att_residual=att_residual)
        # 根据 c3k 标志重新构建 self.m
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )