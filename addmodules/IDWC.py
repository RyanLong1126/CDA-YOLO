import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import autopad

class InceptionDWConv2d(nn.Module):
    """Inception深度卷积模块

    按照通道拆分比例划分四个分支:
    - 恒等映射分支: 直接传递原始特征
    - 3x3方形深度卷积分支: 捕捉常规方形感受野特征
    - 1x11水平带状深度卷积分支: 捕捉横向长距离依赖
    - 11x1垂直带状深度卷积分支: 捕捉纵向长距离依赖
    """

    def __init__(self,
                 in_channels,  # 输入通道数
                 square_kernel_size=3,  # 方形卷积核大小，默认3
                 band_kernel_size=11,  # 带状卷积核大小，默认11
                 branch_ratio=0.125):  # 每个卷积分支的通道占比，默认1/8
        super().__init__()
        # 根据占比计算出每个卷积分支的通道数
        gc = int(in_channels * branch_ratio)
        # 恒等分支: 不进行任何卷积操作，使用剩余通道
        self.identity_channels = in_channels - 3 * gc

        # 3x3方形深度卷积(模拟大核感受野)
        self.conv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2,
                                 groups=gc, bias=False) if gc > 0 else None

        # 1x11水平带状深度卷积(捕获横向上下文)
        self.conv_w = nn.Conv2d(gc, gc, (1, band_kernel_size),
                                padding=(0, band_kernel_size // 2),
                                groups=gc, bias=False) if gc > 0 else None

        # 11x1垂直带状深度卷积(捕获纵向上下文)
        self.conv_h = nn.Conv2d(gc, gc, (band_kernel_size, 1),
                                padding=(band_kernel_size // 2, 0),
                                groups=gc, bias=False) if gc > 0 else None

        # BatchNorm各分支独立
        self.bn_hw = nn.BatchNorm2d(gc) if gc > 0 else None
        self.bn_w = nn.BatchNorm2d(gc) if gc > 0 else None
        self.bn_h = nn.BatchNorm2d(gc) if gc > 0 else None

    def forward(self, x):
        # 按通道拆分
        idx = 0
        # 恒等分支: 直接传递
        id_out = x[:, :self.identity_channels, :, :]
        idx += self.identity_channels

        outputs = [id_out]
        # 方形卷积分支
        if self.conv_hw is not None:
            x_hw = x[:, idx:idx + self.conv_hw.in_channels, :, :]
            out_hw = self.conv_hw(x_hw)
            outputs.append(self.bn_hw(out_hw))
            idx += self.conv_hw.in_channels

        # 水平卷积分支
        if self.conv_w is not None:
            x_w = x[:, idx:idx + self.conv_w.in_channels, :, :]
            out_w = self.conv_w(x_w)
            outputs.append(self.bn_w(out_w))
            idx += self.conv_w.in_channels

        # 垂直卷积分支
        if self.conv_h is not None:
            x_h = x[:, idx:idx + self.conv_h.in_channels, :, :]
            out_h = self.conv_h(x_h)
            outputs.append(self.bn_h(out_h))

        # 拼接所有分支
        return torch.cat(outputs, dim=1)


class C3k2_IDWC(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, shortcut=True,
                 g=1, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 1x1卷积降维，将通道数翻倍
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 1x1卷积输出

        # 构建多分支结构: IDWC + Bottleneck的组合
        self.m = nn.ModuleList(Bottleneck_IDWC(self.c, self.c, shortcut, g, k=(3, 3))
                               for _ in range(n))

    def forward(self, x):
        # 1x1卷积降维
        y = list(self.cv1(x).chunk(2, 1))
        # 多分支特征提取
        y.extend(m(y[-1]) for m in self.m)
        # 拼接后1x1卷积输出
        return self.cv2(torch.cat(y, 1))


class Bottleneck_IDWC(nn.Module):
    """将Bottleneck中的卷积替换为IDWC模块"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5,
                 square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        # InceptionDWConv2d 不改变通道数，输出仍为 c1
        self.cv1 = InceptionDWConv2d(c1, square_kernel_size, band_kernel_size, branch_ratio)
        # 1x1 卷积将 c1 映射到 c2
        self.cv2 = Conv(c1, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Conv(nn.Module):
    """标准卷积块: Conv + BN + SiLU"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))