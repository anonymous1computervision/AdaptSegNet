import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .networks import ResBlock_2018
from .networks import FirstResBlock_2018_SN
from .networks import ResBlock_2018_SN

class XiaoDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=16):
        super(XiaoDiscriminator, self).__init__()
        self.model_pre = []
        # channe = 64
        self.model_pre += [FirstResBlock_2018_SN(num_classes, ndf, downsample=False, use_BN=True)]
        self.model_pre += [ResBlock_2018_SN(ndf, ndf, downsample=False, use_BN=True)]
        # channe = 128
        self.model_pre += [ResBlock_2018_SN(ndf, ndf*2, downsample=False, use_BN=True)]
        self.model_pre += [ResBlock_2018_SN(ndf*2, ndf*2, downsample=False, use_BN=True)]

        # use cGANs with projection
        self.proj_conv =  nn.Conv2d( ndf*2, num_classes, kernel_size=3, stride=1, padding=1)

        self.model_block = []
        # channel = 128
        self.model_block += [ResBlock_2018_SN(ndf*2, ndf*2, downsample=True, use_BN=True)]
        # channel = 256
        self.model_block += [ResBlock_2018_SN(ndf*2, ndf*4, downsample=True, use_BN=True)]
        # channel = 512
        self.model_block += [ResBlock_2018_SN(ndf*4, ndf*8, downsample=True, use_BN=True)]
        # channel = 1024
        self.model_block += [ResBlock_2018_SN(ndf*8, ndf*16, downsample=True, use_BN=True)]

        # use some trick
        self.model_block += [nn.ReLU(inplace=True)]
        self.model_block += [nn.AdaptiveAvgPool2d(ndf*16)]
        self.model_block += [nn.Linear(ndf*16, 1)]

        # create model
        self.model_pre = nn.Sequential(*self.model_pre)
        self.model_block = nn.Sequential(*self.model_block)

        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, label=None):
        if label is None:
            # print("copy x")
            label = x.clone().cuda(0)
        # # sementic label bathc * h * w
        # # value = (0~num_classes)
        # # if label is None:
        # #     x_ =  x.permute(0, 2, 3, 1).detach()
        # #     label = torch.argmax(x_, -1)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # proj_x = self.proj_conv(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # # x = self.conv6(x)
        # # x = self.conv7(x)
        # # x = self.conv8(x)
        # # x = self.conv9(x)
        # x = self.activation(x)
        #
        # # x = self.global_pooling(x)  # global average pooling
        # x = x.sum(2).sum(2)
        # output = self.linear(x)
        # x = self.linear(x)

        # print("label shape = ", label.shape)
        # print("proj_x shape = ", proj_x.shape)
        # pdb.set_trace()

        x = self.model_pre(x)
        proj_x = self.proj_conv(x)
        output = self.model_block(x)
        output += torch.sum(proj_x*label)

        # x += a
        return output
