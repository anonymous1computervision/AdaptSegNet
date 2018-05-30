import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import ResBlockProjection

class XiaoDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=1):
        super(XiaoDiscriminator, self).__init__()
        self.downsample = nn.AvgPool2d(3, stride=2)
        self.activation = nn.ReLU(inplace=True)
        # channel = 64
        self.conv1 = ResBlockProjection(num_classes, ndf, downsample='none', activation="relu")
        # self.conv1 = ResBlockProjection(num_classes, ndf, downsample=self.downsample, activation="relu")
        self.conv2 = ResBlockProjection(ndf, ndf, downsample='none', activation="relu")
        # channel = 128
        # self.conv3 = ResBlockProjection(ndf, ndf*2, downsample=self.downsample, activation="relu")
        self.conv3 = ResBlockProjection(ndf, ndf*2, downsample='none', activation="relu")
        self.conv4 = ResBlockProjection(ndf*2, ndf*2, downsample='none', activation="relu")
        self.conv5 = ResBlockProjection(ndf*2, ndf*2, downsample=self.downsample, activation="relu")
        # channel = 256
        self.conv6 = ResBlockProjection(ndf*2, ndf*4, downsample=self.downsample, activation="relu")
        # channel = 512
        self.conv7 = ResBlockProjection(ndf*4, ndf*8, downsample=self.downsample, activation="relu")
        # channel = 1024
        self.conv8 = ResBlockProjection(ndf*8, ndf*16, downsample=self.downsample, activation="relu")
        self.conv9 = ResBlockProjection(ndf*16, ndf*16, downsample='none', activation="relu")

        self.proj_conv = nn.Conv2d(ndf*2, 1, kernel_size=3, stride=1, padding=1)

        self.global_pooling = nn.AdaptiveAvgPool2d(ndf*16)

        # self.classifier = nn.Conv2d(ndf*16, 1, kernel_size=4, stride=2, padding=1)
        self.linear = nn.Linear(ndf*16, 1)
        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.selu = nn.SELU(inplace=True)

    def forward(self, x, label=None):
        # sementic label bathc * h * w
        # value = (0~num_classes)
        if label is None:
            x_ =  x.permute(0, 2, 3, 1)
            label = torch.argmax(x_, -1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        proj_x = self.proj_conv(x)

        # todo: cgan
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.activation(x)
        x = self.global_pooling(x)  # global average pooling
        x = self.linear(x)
        a = torch.dot(proj_x.view(-1).type(torch.cuda.FloatTensor), label.view(-1).type(torch.cuda.FloatTensor))
        x += a
        return x
