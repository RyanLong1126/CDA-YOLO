import torch
import torch.nn as nn
import torch.nn.functional as F

# 原始 Conv 模块（您提供的实现）
class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = d * (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    """轻量通道注意力模块 (SE)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DualScaleConv(nn.Module):
    """
    双尺度卷积模块 + 通道注意力 + 可选残差。
    原始功能：通道分组 -> 标准卷积 & 空洞卷积 -> 拼接。
    创新点：拼接后添加 SE 注意力，并支持残差连接（c1==c2时有效）。
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        参数:
            use_se: 是否启用通道注意力
            use_residual: 是否启用残差连接（仅当 c1==c2 且 use_residual=True 时生效）
        """
        super().__init__()
        assert k == 3 and s == 1, "DualScaleConv only supports kernel_size=3 and stride=1"
        self.c1, self.c2 = c1, c2


        # 通道分组（保持与原始一致）
        c1_1 = c1 // 2
        c1_2 = c1 - c1_1
        c2_1 = c2 // 2
        c2_2 = c2 - c2_1
        self.c1_1, self.c1_2 = c1_1, c1_2

        # 双尺度卷积分支
        self.conv1 = Conv(c1_1, c2_1, 3, 1, act=act)          # dilation=1
        self.conv2 = Conv(c1_2, c2_2, 3, 1, d=2, act=act)    # dilation=2


    def forward(self, x):
        # 原始双尺度分支
        x1, x2 = torch.split(x, [self.c1_1, self.c1_2], dim=1)
        out1 = self.conv1(x1)
        out2 = self.conv2(x2)
        out = torch.cat([out1, out2], dim=1)   # shape: [B, c2, H, W]


        return out