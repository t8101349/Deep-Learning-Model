import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3, padding=1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)


class UNetplusplus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512],deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)
        self.up_sample = lambda in_ch, out_ch: nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        self.conv00 = DoubleConv(in_channels, features[0])
        self.conv10 = DoubleConv(features[0], features[1])
        self.conv20 = DoubleConv(features[1], features[2])
        self.conv30 = DoubleConv(features[2], features[3])
        self.conv40 = DoubleConv(features[3], features[3]*2)
        
        self.conv01 = DoubleConv(features[0] + features[1], features[0])
        self.conv11 = DoubleConv(features[1] + features[2], features[1])
        self.conv21 = DoubleConv(features[2] + features[3], features[2])
        self.conv31 = DoubleConv(features[3] + features[3]*2, features[3])

        self.conv02 = DoubleConv(features[0]*2 + features[1], features[0])
        self.conv12 = DoubleConv(features[1]*2 + features[2], features[1])
        self.conv22 = DoubleConv(features[2]*2 + features[3], features[2])

        self.conv03 = DoubleConv(features[0]*3 + features[1], features[0])
        self.conv13 = DoubleConv(features[1]*3 + features[2], features[1])

        self.conv04 = DoubleConv(features[0]*4 + features[1], features[0])

        self.final = nn.Conv2d(features[0], out_channels, kernal_size = 1)
        if self.deep_supervision:
            self.final1 = nn.Conv2d(features[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(features[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))
        x40 = self.conv40(self.pool(x30))

        # Decode r blocks with nested skip
        x01 = self.conv01(torch.cat([x00,self.up_sample(x10.size(1),x00.size(1))(x10)], dim = 1))
        x11 = self.conv11(torch.cat([x10,self.up_sample(x20.size(1),x10.size(1))(x20)], dim = 1))
        x21 = self.conv21(torch.cat([x20,self.up_sample(x30.size(1),x20.size(1))(x30)], dim = 1))
        x31 = self.conv31(torch.cat([x30,self.up_sample(x40.size(1),x30.size(1))(x40)], dim = 1))
       

        x02 = self.conv02(torch.cat([x00,x01,self.up_sample(x11.size(1),x01.size(1))(x11)], dim = 1))
        x12 = self.conv12(torch.cat([x10,x11,self.up_sample(x21.size(1),x11.size(1))(x21)], dim = 1))
        x22 = self.conv22(torch.cat([x20,x21,self.up_sample(x31.size(1),x21.size(1))(x31)], dim = 1))

        x03 = self.conv03(torch.cat([x00,x01,x02,self.up_sample(x12.size(1),x02.size(1))(x12)], dim = 1))
        x13 = self.conv13(torch.cat([x10,x11,x12,self.up_sample(x22.size(1),x12.size(1))(x22)], dim = 1))

        x04 = self.conv04(torch.cat([x00,x01,x02,x03,self.up_sample(x13.size(1),x03.size(1))(x13)], dim = 1))

        if self.deep_supervision:
            return [
                self.final(x01),
                self.final1(x02),
                self.final2(x03),
                self.final3(x04),
            ]
        else:
            return self.final(x04)