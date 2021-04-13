import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class MobileUnet(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 2),
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, out_ch=5):
        super(MobileUnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers1 = self.make_layers(in_planes=32,config=self.cfg[0])
        self.layers2 = self.make_layers(in_planes=16,config=self.cfg[1])
        self.layers3 = self.make_layers(in_planes=24,config=self.cfg[2])
        self.layers4 = self.make_layers(in_planes=32,config=self.cfg[3])
        self.layers5 = self.make_layers(in_planes=64,config=self.cfg[4])
        self.layers6 = self.make_layers(in_planes=96,config=self.cfg[5])

        self.up6 = nn.ConvTranspose2d(160, 96, 2, stride=2)
        self.conv6 = DoubleConv(256, 96)
        self.up7 = nn.ConvTranspose2d(96, 32, 2, stride=2)
        self.conv7 = DoubleConv(64, 32)
        self.up8 = nn.ConvTranspose2d(32, 24, 2, stride=2)
        self.conv8 = DoubleConv(48, 24)
        self.up9 = nn.ConvTranspose2d(24, 16, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.up10 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.conv10 = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        c0 = F.relu(self.bn1(self.conv1(x)))
        c1 = self.layers1(c0)
        c2 = self.layers2(c1)
        c3 = self.layers3(c2)
        c4 = self.layers4(c3)
        c5 = self.layers5(c4)
        c6 = self.layers6(c5)
        up_6 = self.up6(c6)
        merge6 = torch.cat([up_6, c5, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1, c0], dim=1)
        c9 = self.conv9(merge9)
        up_10 = self.up10(c9)
        c10 = self.conv10(up_10)
        out = nn.Sigmoid()(c10)
        return out


    def make_layers(self,in_planes,config):
        layers = []
        expansion, out_planes, num_blocks, stride = config
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(
                Block(in_planes, out_planes, expansion, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    """def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(
                    Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)"""




if __name__ == "__main__":
    net = MobileUnet(5).cuda()
    summary(net, (3,224,224))

