import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


class DeformableConv2d(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            *,
            offset_groups=1,
            with_mask=False
    ):
        super().__init__()
        assert in_dim % groups == 0, f"无法分组卷积"
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv_ = nn.Conv2d(in_dim, out_dim, kernel_size, groups=groups)
        self.weight = nn.Parameter(self.conv_.weight)
        if bias:
            self.bias = nn.Parameter(self.conv_.bias)
        else:
            self.bias = None

        self.with_mask = with_mask
        if with_mask:
            # batch_size, (2+1) * offset_groups * kernel_height * kernel_width, out_height, out_width
            self.param_generator = nn.Conv2d(in_dim, 3 * offset_groups * kernel_size * kernel_size, 3, 1, 1)
        else:
            self.param_generator = nn.Conv2d(in_dim, 2 * offset_groups * kernel_size * kernel_size, 3, 1, 1)

    def forward(self, x):
        if self.with_mask:
            oh, ow, mask = self.param_generator(x).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:
            offset = self.param_generator(x)
            mask = None
        x = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            mask=mask,
        )
        return x


class DeformConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DeformConv, self).__init__(
            DeformableConv2d(in_channels, out_channels, kernel_size=3, padding=1, with_mask=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DoubleDeform(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            DeformConv(in_channels, out_channels),
            DeformConv(out_channels, out_channels)
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class DownDeform(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleDeform(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, deform=False):
        super(Up, self).__init__()
        # 原论文采用的是转置卷积，我们一般用双线性插值
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if deform:
                self.conv = DoubleDeform(in_channels, out_channels)
            else:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:  # 采用转置卷积的通道数会减少一半
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            if deform:
                self.conv = DoubleDeform(in_channels, out_channels)
            else:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # 为了防止maxpooling后得到的图片尺寸向下取整，不是整数倍
        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            nn.Sigmoid()
        )


class DU_Net(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 16):
        super(DU_Net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = DownDeform(base_c, base_c * 2)
        self.down2 = DownDeform(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 16 // factor, base_c * 16 // factor)
        self.up1 = Up(base_c * 8 + base_c * 16 // factor, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8 // factor + base_c * 4, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4 // factor + base_c * 2, base_c * 2 // factor, bilinear, deform=True)
        self.up4 = Up(base_c * 2 // factor + base_c, base_c, bilinear, deform=True)
        self.out_conv = OutConv(base_c + in_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(torch.cat((x, input), dim=1))

        return logits
