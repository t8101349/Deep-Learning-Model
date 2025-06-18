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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels , out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True),
        )
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=[64,128,256,512]):
        super().__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(in_channels, in_channels*2)

        for feature in features[::-1]:
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, 2, 2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels , 1)

    def forward(self,x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, (2,2))
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):   #一次讀兩層
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](x)
        
        return self.final_conv(x)


model = UNet()
toy_data = torch.ones((16,3,240,160))
output = model(toy_data)
print(output.shape)

#model.cuda()

class Test:
    def __len__(self):
        return 100
    def __getitem__(self, i):
        return f"This is the {i}th datapoint."
    
test = Test()
print(len(test))


'''
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
            x = self.ups[index-1](x)

        x = self.final_conv(x)
        # x = F.log_softmax(x)
        return x
'''