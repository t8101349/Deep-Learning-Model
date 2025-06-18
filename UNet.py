import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.in_ch = in_channels
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(double_conv(self.in_ch, feature))
            self.downs.append(self.down_sample)
            self.in_ch = feature

        self.bottleneck = double_conv(self.in_ch , self.in_ch*2)

        for feature in features[::-1]:
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(double_conv(feature*2, feature))


        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connect = []

        for index in range(0,len(self.downs),2):
            x = self.downs[index](x)
            skip_connect.append(x)
            x = self.downs[index+1](x)

        x = self.bottleneck(x)

        for index in range(0, len(self.ups), 2):
            x = self.ups[index](x)
            skip_ = skip_connect.pop()
            if x.shape != skip_.shape:
                x = F.interpolate(x, size=skip_.shape[2:])
            x = torch.cat((skip_, x), dim=1)
            x = self.ups[index+1](x)

        x = self.final_conv(x)
        # x = F.log_softmax(x)
        return x



if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 572, 572)
    y = model(x)
    print(y.shape)  # -> 應為 [1, 1, 572, 572] 或接近尺寸
