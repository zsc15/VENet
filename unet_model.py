""" Full assembly of the parts to form the complete network """

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        #####
        self.outc = OutConv(64, n_classes)
        self.conv2 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        y = self.up4(x, x1)
        logits = self.outc(y)
        x_tanh = self.tanh(logits)
        x_seg = self.conv2(y)
        return x_tanh, x_seg

class Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        #####
        # self.outc = OutConv(64, n_classes)
        self.conv2 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)
        # self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        y = self.up4(x, x1)
        y = self.conv2(y)
        # x_tanh = self.tanh(logits)
        # x_seg = self.sigmoid(y)
        return y

class UNet2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True, compute_sdm=False):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.compute_sdm = compute_sdm
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        #####
        if self.compute_sdm:
            self.outc = OutConv(64, n_classes)
            self.tanh = nn.Tanh()
        self.conv2 = nn.Conv2d(64, 2, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        y = self.up4(x, x1)
        if self.compute_sdm:
            logits = self.outc(y)
            x_tanh = self.tanh(logits)
            x_seg = self.conv2(y)
            return x_tanh, x_seg
        else:
            return y, self.conv2(y)

class VENet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True, compute_sdm=False):
        super(VENet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.compute_sdm = compute_sdm
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        #####
        if self.compute_sdm:
            self.outc = OutConv(64, n_classes)
            self.tanh = nn.Tanh()
        self.conv2_1 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        y = self.up4(x, x1)
        return torch.cat((self.conv2_1(y),self.conv2_2(y)), dim=1)

class UNet2_with_contour(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(UNet2_with_contour, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        ######
        self.up_contour1 = Up(1024, 512 // factor, bilinear)
        self.up_contour2 = Up(512, 256 // factor, bilinear)
        self.up_contour3 = Up(256, 128 // factor, bilinear)
        self.up_contour4 = Up(128, 64, bilinear)
        ######
        self.out_contour = OutConv(64, 1)
        self.conv2 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)
        # self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5_f = self.down4(x4)
        x = self.up1(x5_f, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        y = self.up4(x, x1)
        x_seg = self.conv2(y)
        #####sdm
        z = self.up_contour1(x5_f, x4)
        z = self.up_contour2(z, x3)
        z = self.up_contour3(z, x2)
        z = self.up_contour4(z, x1)
        x_contour = self.out_contour(z)
        # x_tanh = self.tanh(logits)
        return x_contour, x_seg

class UNet_UAMT(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(UNet_UAMT, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        #####
        self.outc = OutConv(64, n_classes)
        self.conv2 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)
        # self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        y = self.up4(x, x1)
        # logits = self.outc(y)
        # x_tanh = self.tanh(logits)
        x_seg = self.conv2(y)
        return x_seg