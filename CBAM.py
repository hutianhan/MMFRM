
# ------------------#
# ResBlock+CBAM
# ------------------#
import torch
import torch.nn as nn
import torchvision


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        k=self.sigmoid(avgout + maxout)
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        #print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out

        return out

class CBAM_Spatial(nn.Module):
    def __init__(self):
        super(CBAM_Spatial, self).__init__()
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.spatial_attention(x) * x
        #print("CBAM_Spatial", self.spatial_attention(x).shape,x.shape)
        return out

class CBAM_channel(nn.Module):
    def __init__(self, channel):
        super(CBAM_channel, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
    def forward(self, x):
        out = self.channel_attention(x) * x
        #print("CBAM_channel",self.channel_attention(x).shape,x.shape)
        return out


class SE_Block(nn.Module):                         # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y= x
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)*y
        return out


if __name__=="__main__":
    m1 = CBAM_Spatial()
    m2 = CBAM_channel(channel=32)
    m3=   SE_Block(in_planes=32)
    m4=CBAM(channel=32)
    input = torch.randn(1, 32, 64, 64)
    out1 = m1(input)
    out2 = m2(input)
    out3=  m3(input)
    out4 = m4(input)
    print(out1.shape,out2.shape,out3.shape,out4.shape)

'''
class css(nn.Module):
    def __init__(self,):
        super(css, self).__init__()
        self.l=nn.LeakyReLU()
        #self.l = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        k=self.l(x)
        print(k)
        k=self.sigmoid(x)
        return k

x=torch.randn(2,1,2,2)
net=css()
y=net(x)
print(x)
print(y)
'''


'''
class ResBlock_CBAM(nn.Module):
    def __init__(self,in_places, places, stride=1,downsampling=False, expansion = 1):
        super(ResBlock_CBAM,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )
        self.cbam = CBAM(channel=places*self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        print(x.shape)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


model = ResBlock_CBAM(in_places=32, places=32)
print(model)

input = torch.randn(1, 32, 64, 64)
out = model(input)
print(out.shape)
'''