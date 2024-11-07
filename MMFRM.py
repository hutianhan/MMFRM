import torch
import torch.nn as nn
from torch.nn import functional as F

from CBAM import *
class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)

class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)

class Conv_Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class TCFFM(nn.Module):
    def __init__(self, channel1,channel2):
        super(TCFFM, self).__init__()

        self.conv1 = nn.Conv2d(channel1, channel1, kernel_size=3, stride=1, padding=1)
        self.cbam1 = CBAM(channel1)
        self.conv2 = nn.Conv2d(channel2, channel2, kernel_size=3, stride=1, padding=1)
        self.cbam2 = CBAM(channel2)
        self.conv3 = nn.Conv2d(channel1+channel2, channel1+channel2, kernel_size=3, stride=1, padding=1)
        self.cbam3 = CBAM(channel1+channel2)

    def forward(self, x,y):
        x=self.conv1(x)
        x=self.cbam1(x)
        y = self.conv2(y)
        y = self.cbam2(y)
        z=torch.cat([x, y], dim=1)
        z=self.conv3(z)
        z=self.cbam3(z)
        return z

class MMFRM(nn.Module):
    def __init__(self,  num_classes=7, drop_rate=0.0):
        super(MMFRM, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        # self.layer3 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
        #                             RestNetBasicBlock(512, 512, 1))
        #
        # self.layer4 = nn.Sequential(RestNetDownBlock(512, 1024, [2, 1]),
        #                             RestNetBasicBlock(1024, 1024, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.drop_rate = drop_rate


        self.fc = nn.Linear(512, num_classes)
        self.alpha = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

        self.cov1 =  Conv_Block(128 +64,128)
        self.cov2 = Conv_Block(256 + 128, 256)
        self.cov3 = Conv_Block(512 + 512, 512)

        self.TCFMM1 = TCFFM(128, 64)
        self.TCFMM2= TCFFM(256,128)
        self.TCFMM3 = TCFFM(512, 512)

    def forward(self, x,feature1,feature2,feature3):

        out = self.conv1(x)
        out = self.layer1(out)

        out = self.layer2(out)#[B, 128, 28, 28]
        # out = torch.cat([out, feature1], dim=1)  #c:128 ,64
        out=self.TCFMM1(out, feature1)
        out= self.cov1(out)

        out = self.layer3(out)#[B, 256, 14, 14]
        # out = torch.cat([out, feature2], dim=1)  #c:256 ,128
        out = self.TCFMM2(out, feature2)
        out= self.cov2(out)

        out = self.layer4(out)#[B, 512, 7, 7]
        # out = torch.cat([out, feature3], dim=1) #c:512 ,512
        out = self.TCFMM3(out, feature3)
        out= self.cov3(out)

        if self.drop_rate > 0:
            out = nn.Dropout(self.drop_rate)(out)
        out=self.avgpool(out)
        out = out.view(out.size(0), -1)

        attention_weights = self.alpha(out)
        out = attention_weights * self.fc(out)

        # out = self.fc(out)
        return attention_weights, out





