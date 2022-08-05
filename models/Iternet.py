import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # 原论文采用的是转置卷积，我们一般用双线性插值
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:  # 采用转置卷积的通道数会减少一半
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # 为了防止maxpooling后得到的图片尺寸向下取整，不是整数倍
        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)
        return out


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            nn.Sigmoid()
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 bilinear: bool = False,
                 base_c: int = 32):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return x1, x, logits


class Min_UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 32,
                 num_classes: int = 1,
                 bilinear: bool = False,
                 base_c: int = 64):
        super(Min_UNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.down1 = Down(base_c, base_c * 2)

        factor = 2 if bilinear else 1
        self.down2 = Down(base_c * 2, base_c * 4 // factor)
        self.up1 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up2 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.up1(x2, x1)
        x = self.up2(x3, x)

        logits = self.out_conv(x)

        return x, logits


class IterNet(nn.Module):
    def __init__(self, in_channels=3, num_class=1, base_c=32, iteration=3):
        super().__init__()
        self.iter = iteration
        self.base_c = base_c
        self.in_channels = in_channels
        self.num_class = num_class
        self.main_unet = UNet(in_channels, num_class, bilinear=False, base_c=base_c)
        self.min_unet = Min_UNet(base_c, num_class, bilinear=False, base_c=base_c)
        self.conv = DoubleConv(self.base_c, self.base_c)
        self.conv2 = nn.ModuleList(
            [nn.Conv2d(32 * (i + 2), self.base_c, kernel_size=1) for i in range(self.iter)])

    def forward(self, x):
        outs = []
        x1, x1_last, out1 = self.main_unet(x)
        # 用来存放out[i]
        outs.append(out1)
        # 用来feed进子网络
        input = [x1_last]
        # 用来进行cat连接的输出
        concat = [x1]
        for k in range(self.iter):
            # 只有那个1*1卷积层不是共享的
            x_input = self.conv(input[-1])
            concat.append(x_input)
            x_cat = torch.cat(concat, dim=1)
            x_cat = self.conv2[k](x_cat)
            x_last, out = self.min_unet(x_cat)
            outs.append(out)
            input.append(x_last)

        return outs
        # return [out1, out2, out3, out4]
