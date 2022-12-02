import torch
from torch import nn


class SPConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, alpha: float = 0.5):
        super(SPConv2D, self).__init__()
        assert 0 <= alpha <= 1
        self.alpha = alpha

        self.in_rep_channels = int(in_channels * self.alpha)
        self.out_rep_channels = int(out_channels * self.alpha)
        self.out_channels = out_channels
        self.stride = stride

        self.represent_gp_conv = nn.Conv2d(in_channels=self.in_rep_channels,
                                           out_channels=self.out_channels,
                                           stride=self.stride,
                                           kernel_size=3,
                                           padding=1,
                                           groups=2)
        self.represent_pt_conv = nn.Conv2d(in_channels=self.in_rep_channels,
                                           out_channels=out_channels,
                                           kernel_size=1)

        self.redundant_pt_conv = nn.Conv2d(in_channels=in_channels - self.in_rep_channels,
                                           out_channels=out_channels,
                                           kernel_size=1)

        self.avg_pool_s2_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_s2_3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.avg_pool_add_1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_add_3 = nn.AdaptiveAvgPool2d(1)

        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        self.group = int(1 / self.alpha)

    def forward(self, x):
        batch_size = x.size()[0]

        x_3x3 = x[:, :self.in_rep_channels, ...]
        x_1x1 = x[:, self.in_rep_channels:, ...]
        rep_gp = self.represent_gp_conv(x_3x3)

        if self.stride == 2:
            x_3x3 = self.avg_pool_s2_3(x_3x3)
        rep_pt = self.represent_pt_conv(x_3x3)
        rep_fuse = rep_gp + rep_pt
        rep_fuse = self.bn1(rep_fuse)
        rep_fuse_ration = self.avg_pool_add_3(rep_fuse).squeeze(dim=3).squeeze(dim=2)

        if self.stride == 2:
            x_1x1 = self.avg_pool_s2_1(x_1x1)

        red_pt = self.redundant_pt_conv(x_1x1)
        red_pt = self.bn2(red_pt)
        red_pt_ratio = self.avg_pool_add_1(red_pt).squeeze(dim=3).squeeze(dim=2)

        out_31_ratio = torch.stack((rep_fuse_ration, red_pt_ratio), 2)
        out_31_ratio = nn.Softmax(dim=2)(out_31_ratio)

        out_mul_1 = red_pt * (out_31_ratio[:, :, 1].view(batch_size, self.out_channels, 1, 1).expand_as(red_pt))
        out_mul_3 = rep_fuse * (out_31_ratio[:, :, 0].view(batch_size, self.out_channels, 1, 1).expand_as(rep_fuse))

        return out_mul_1 + out_mul_3