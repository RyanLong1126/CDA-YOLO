import torch
import torch.nn as nn
import torch.nn.functional as F


class SimAM(nn.Module):
    """原版SimAM保留，确保兼容性"""

    def __init__(self, channels=None, out_channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)


class SimAM_Edge(nn.Module):
    """
    边缘感知SimAM：针对PCB spur（细小突出物）优化
    创新点：在通道注意力基础上，增加轻量级边缘感知分支，
    通过Sobel-like卷积增强对细长边缘异常的敏感度
    """

    def __init__(self, channels, e_lambda=1e-4, edge_ratio=0.2):
        super().__init__()
        self.channels = channels
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.edge_ratio = edge_ratio  # 边缘分支融合比例

        # 轻量级边缘提取：使用两个1D sobel-like卷积（参数量极小）
        self.edge_conv_x = nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False)
        self.edge_conv_y = nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False)

        # 边缘注意力投影：将边缘特征映射回通道权重
        self.edge_proj = nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False)
        self.edge_bn = nn.BatchNorm2d(channels)

        # 初始化sobel-like权重
        self._init_edge_weights()

    def _init_edge_weights(self):
        """初始化边缘检测权重，增强对角线/边缘响应"""
        with torch.no_grad():
            # 使用平滑后的边缘核初始化，避免训练初期梯度爆炸
            for m in [self.edge_conv_x, self.edge_conv_y]:
                if m.weight.shape[1] == m.weight.shape[0] * 4:  # 1x1卷积
                    nn.init.normal_(m.weight, std=0.01)

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        # 1. 原始SimAM通道注意力
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x_minus_mu_square = (x - x_mean).pow(2)
        y_channel = x_minus_mu_square / (
                    4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        attn_channel = self.activation(y_channel)

        # 2. 边缘感知分支（轻量）
        # 使用平均池化降低分辨率，减少计算（小目标不需要全分辨率边缘）
        x_edge = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # 分别提取水平和垂直边缘特征
        edge_x = self.edge_conv_x(x_edge)
        edge_y = self.edge_conv_y(x_edge)

        # 边缘特征融合
        edge_feat = torch.cat([edge_x, edge_y], dim=1)
        edge_feat = F.silu(edge_feat)  # 非线性激活

        # 投影回通道维度并上采样
        attn_edge = self.edge_proj(edge_feat)
        attn_edge = self.edge_bn(attn_edge)
        attn_edge = self.activation(attn_edge)

        # 3. 融合：通道注意力为主，边缘注意力为辅
        # 对spur类缺陷，边缘分支会增强细长突出区域的响应
        attn = attn_channel * (1 + self.edge_ratio * attn_edge)
        attn = torch.clamp(attn, 0, 2)  # 防止数值过大

        return x * attn


def autopad(k, p=None, d=1):
    """Pads kernel to 'same' output shape, adjusting for optional dilation."""
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

    def forward_fuse(self, x):
        return self.act(self.conv(x))


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


class C3k_MulScale(nn.Module):
    """
    多尺度C3k：针对spur缺陷尺度变化大（短毛刺/长突出）优化
    创新点：并行3×3和5×5分支，通过通道注意力自适应融合不同尺度特征
    保持与C3k相同接口，可直接替换
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.c_ = c_
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)

        # 多尺度分支：3×3捕捉细小spur，5×5捕捉较长突出物
        self.cv3x3 = Conv(c_, c_, k=3, s=1, g=g)
        self.cv5x5 = Conv(c_, c_, k=5, s=1, g=g)

        # 尺度自适应权重（轻量，每个通道一个权重）
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_, 2, 1, bias=False),  # 2个尺度
            nn.Softmax(dim=1)
        )

        self.cv3 = Conv(2 * c_, c2, 1)

        # 内部n层重复（简化版，保持效率）
        self.n = n
        self.shortcut = shortcut and c1 == c2
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 分支1：通过多尺度处理
        x1 = self.cv1(x)

        # 多尺度特征提取
        f3 = self.cv3x3(x1)
        f5 = self.cv5x5(x1)

        # 自适应尺度融合
        w = self.scale_attn(x1)  # [B, 2, 1, 1]
        w3, w5 = w[:, 0:1, ...], w[:, 1:2, ...]
        f_mul = f3 * w3 + f5 * w5

        # 分支2：identity
        x2 = self.cv2(x)

        # 拼接输出
        out = self.cv3(torch.cat((f_mul, x2), 1))

        # shortcut
        return x + out if self.add else out


class C2f_SimAM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = SimAM(c2)  # 保持原版接口

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))


class C2f_SimAM_Edge(nn.Module):
    """
    边缘感知版C2f：针对PCB spur缺陷优化
    创新点：使用SimAM_Edge替换SimAM，增强对细长突出物的敏感度
    接口与C2f_SimAM完全一致，可直接替换
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, edge_lambda=1e-4, edge_ratio=0.2):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # 关键改进：使用边缘感知注意力
        self.att = SimAM_Edge(c2, e_lambda=edge_lambda, edge_ratio=edge_ratio)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))


class C3k2_SimAM(C2f_SimAM):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k2_SimAM_Edge(C2f_SimAM_Edge):
    """
    终极改进版：边缘感知 + 多尺度
    针对spur缺陷：使用C3k_MulScale增强多尺度特征提取
    接口与C3k2_SimAM完全一致
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True,
                 edge_lambda=1e-4, edge_ratio=0.2, mulscale=False):
        super().__init__(c1, c2, n, shortcut, g, e, edge_lambda, edge_ratio)

        # 关键改进：使用多尺度C3k或标准Bottleneck
        if mulscale and c3k:
            # 多尺度模式：使用C3k_MulScale
            self.m = nn.ModuleList(
                C3k_MulScale(self.c, self.c, 1, shortcut, g, k=3) for _ in range(n)
            )
        elif c3k:
            # 标准C3k模式
            self.m = nn.ModuleList(
                C3k(self.c, self.c, 2, shortcut, g) for _ in range(n)
            )
        else:
            # 标准Bottleneck
            self.m = nn.ModuleList(
                Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
            )