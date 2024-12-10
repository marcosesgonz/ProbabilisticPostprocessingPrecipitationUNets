import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            #nn.PReLU(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            #nn.PReLU()
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is NCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1) #En Unet original: kernel_size = 1 y padding=0. A lo mejor no conviene añadir padding tan tarde
        self.conv_relu = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 3,padding = 1),
                                       nn.Softplus())#nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv_relu(x)
    

""" Full assembly of the parts to form the complete network """
class UNet(nn.Module):
    def __init__(self, n_inp_channels, n_outp_channels, red_factor = 1, bilinear=False, bottleneck = False):
        super(UNet, self).__init__()
        self.n_channels = n_inp_channels
        self.n_outp_channels = n_outp_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_inp_channels, 32//red_factor))
        self.down1 = (Down(32//red_factor, 64//red_factor))
        self.down2 = (Down(64//red_factor, 128//red_factor))
        self.down3 = (Down(128//red_factor, 256//red_factor)) 
        factor = 2 if bilinear else 1
        #self.down4 = (Down(512, 1024 // factor))
        if bottleneck:
            self.middle = (DoubleConv(256//red_factor,256//red_factor)) #added by me
        self.up1 = (Up(256//red_factor, (128 //red_factor) // factor, bilinear)) 
        self.up2 = (Up(128//red_factor, (64 //red_factor) // factor, bilinear))
        self.up3 = (Up(64//red_factor, (32//red_factor) // factor, bilinear))
        #self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(32//red_factor, n_outp_channels))
        self.bottleneck = bottleneck
        
    def forward(self, x):
        #print(f'x {x.size()}')
        x1 = self.inc(x)
        #print(f'x1 {x1.size()}')
        x2 = self.down1(x1)
        #print(f'x2 {x2.size()}')
        x3 = self.down2(x2)
        #print(f'x3 {x3.size()}')
        x4 = self.down3(x3)
        #print(f'x4 {x4.size()}')
        if self.bottleneck:
            x5 = self.middle(x4)
            x = self.up1(x5,x3)
        else:
            x = self.up1(x4,x3)
        #print(x5.size())
        #print(f'x {x.size()}')
        x = self.up2(x, x2)
        #print(f'x {x.size()}')
        x = self.up3(x, x1)
        #print(f'x {x.size()}')
        logits = self.outc(x) #+ 1e-4 #To avoid 0 values
        #print(f'outp {logits.size()}')
        return logits
    """
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.middle = torch.utils.checkpoint(self.middle)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.outc = torch.utils.checkpoint(self.outc)
    """
