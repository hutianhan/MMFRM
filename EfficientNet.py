import torch
import torch.nn as nn

class ConvBNSiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups: int = 1,
                 activation: nn.Module = nn.Identity()):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
        )

    def forward(self, x):
        return self.conv(x)


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=in_channels, out_features=int(in_channels // reduction), bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=int(in_channels // reduction), out_features=in_channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        out = self.conv(x).view(b, c, 1, 1)
        return x * out.expand_as(x)


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, channels_factor, downsample: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tmp_channels = self.in_channels * channels_factor
        self.stride = 1 if not downsample else 2
        self.conv = nn.Sequential(
            ConvBNSiLU(in_channels=self.in_channels, out_channels=self.tmp_channels, kernel_size=1, stride=1, padding=0,
                       activation=nn.SiLU(inplace=True)),
            ConvBNSiLU(in_channels=self.tmp_channels, out_channels=self.tmp_channels, kernel_size=3, stride=self.stride,
                       padding=1, groups=self.tmp_channels, activation=nn.SiLU(inplace=True)),
            SEModule(in_channels=self.tmp_channels, reduction=4),
            ConvBNSiLU(in_channels=self.tmp_channels, out_channels=self.out_channels, kernel_size=1, stride=1,
                       padding=0),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.in_channels == self.out_channels: return out + x
        return out


class EfficientNetB0(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        CFG = [1, 2, 2, 3, 3, 4, 1]
        self.conv = ConvBNSiLU(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.mbconv1_1 = self._make_layers(CFG[0], in_channels=32, out_channels=16, channels_factor=1, downsample=False)
        self.mbconv6_2 = self._make_layers(CFG[1], in_channels=16, out_channels=24, channels_factor=6, downsample=True)
        self.mbconv6_3 = self._make_layers(CFG[2], in_channels=24, out_channels=40, channels_factor=6, downsample=True)
        self.mbconv6_4 = self._make_layers(CFG[3], in_channels=40, out_channels=80, channels_factor=6, downsample=True)
        self.mbconv6_5 = self._make_layers(CFG[4], in_channels=80, out_channels=112, channels_factor=6,
                                           downsample=False)
        self.mbconv6_6 = self._make_layers(CFG[5], in_channels=112, out_channels=192, channels_factor=6,
                                           downsample=True)
        self.mbconv6_7 = self._make_layers(CFG[6], in_channels=192, out_channels=320, channels_factor=6,
                                           downsample=False)
        self.fc = nn.Sequential(
            ConvBNSiLU(in_channels=320, out_channels=1280, kernel_size=1, stride=1, padding=0,
                       activation=nn.SiLU(inplace=True)),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=1280, out_features=n_classes, bias=True)
        )

    def _make_layers(self, num_layers, in_channels, out_channels, channels_factor, downsample):
        layers = [MBConv(in_channels=in_channels, out_channels=out_channels, channels_factor=channels_factor,
                         downsample=downsample)]
        for _ in range(num_layers - 1):
            layers.append(MBConv(in_channels=out_channels, out_channels=out_channels, channels_factor=channels_factor))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        #print('# Conv output shape:', x.shape)
        x = self.mbconv1_1(x)
        #print('# MBConv1_1 output shape:', x.shape)
        x = self.mbconv6_2(x)
        #print('# MBConv6_2 output shape:', x.shape)
        x = self.mbconv6_3(x)
        #print('# MBConv6_3 output shape:', x.shape)
        x = self.mbconv6_4(x)
        #print('# MBConv6_4 output shape:', x.shape)
        x = self.mbconv6_5(x)
        #print('# MBConv6_5 output shape:', x.shape)
        x = self.mbconv6_6(x)
        #print('# MBConv6_6 output shape:', x.shape)
        x = self.mbconv6_7(x)
        #print('# MBConv6_7 output shape:', x.shape)
        x = self.fc(x)
        #print('# FC output shape:', x.shape)
        return x

