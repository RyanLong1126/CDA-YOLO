import torch
import torch.nn as nn

__all__ = ['BiFPN_Concat']

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
    def forward_fuse(self, x):
        return self.act(self.conv(x))

class BiFPN_Concat(nn.Module):
    def __init__(self, c1, c2):
        """c1: 统一通道数（输入调整后的通道）; c2: 最终输出通道"""
        super().__init__()
        self.c1 = c1      # 加权求和前的统一通道数
        self.c2 = c2      # 最终输出通道数
        self.epsilon = 0.0001
        self.act = nn.ReLU()
        self.conv = Conv(c1, c2, 1, 1, 0)   # 最后一步卷积

        # 以下属性在第一次 forward 时动态创建
        self.adjust_convs = nn.ModuleList()  # 每个输入对应的 1x1 卷积
        self.weight_params = None             # 可学习的权重

    def forward(self, x):
        # x 是一个 list，长度 = 2 或 3
        n = len(x)
        device = x[0].device

        # 首次运行：根据输入数量创建 adjust_convs 和 weight_params
        if len(self.adjust_convs) != n:
            self.adjust_convs = nn.ModuleList()
            for i in range(n):
                in_ch = x[i].shape[1]          # 原始输入通道数
                self.adjust_convs.append(Conv(in_ch, self.c1, 1, 1, 0))
            if n == 2:
                self.weight_params = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            elif n == 3:
                self.weight_params = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
            else:
                raise ValueError("BiFPN_Concat only supports 2 or 3 inputs")
            self.to(device)

        # 1. 用 1x1 卷积将每个输入调整到统一通道 c1
        adjusted = [conv(xi) for conv, xi in zip(self.adjust_convs, x)]

        # 2. 加权求和
        w = self.weight_params
        weights = w / (torch.sum(w, dim=0) + self.epsilon)
        if n == 2:
            weighted_sum = weights[0] * adjusted[0] + weights[1] * adjusted[1]
        else:
            weighted_sum = weights[0] * adjusted[0] + weights[1] * adjusted[1] + weights[2] * adjusted[2]

        # 3. 激活 + 最终卷积
        out = self.conv(self.act(weighted_sum))
        return out