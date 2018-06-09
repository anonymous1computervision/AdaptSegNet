import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .networks import ResBlock_2018
from .networks import FirstResBlock_2018_SN
from .networks import ResBlock_2018_SN
from .networks import SpectralNorm

class XiaoDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=16):
        super(XiaoDiscriminator, self).__init__()
        self.model_pre = []
        # channe = 64
        self.model_pre += [FirstResBlock_2018_SN(num_classes, ndf, downsample=True, use_BN=False)]
        self.model_pre += [ResBlock_2018_SN(ndf, ndf, downsample=False, use_BN=False)]
        # channe = 128
        self.model_pre += [ResBlock_2018_SN(ndf, ndf*2, downsample=True, use_BN=False)]
        self.model_pre += [ResBlock_2018_SN(ndf*2, ndf*2, downsample=False, use_BN=False)]
        # channel = 128
        self.model_pre += [ResBlock_2018_SN(ndf*2, ndf*2, downsample=True, use_BN=False)]
        # channel = 256
        self.model_pre += [ResBlock_2018_SN(ndf*2, ndf*4, downsample=True, use_BN=False)]
        # channel = 512
        self.model_pre += [ResBlock_2018_SN(ndf*4, ndf*8, downsample=True, use_BN=False)]

        # use cGANs with projection
        num_classes = 3
        self.proj_conv = SpectralNorm(nn.Conv2d(ndf * 8, num_classes, kernel_size=3, stride=1, padding=1))

        self.model_block = []

        # channel = 1024
        self.model_block += [ResBlock_2018_SN(ndf*8, ndf*16, downsample=True, use_BN=False)]
        # use some trick
        self.model_block += [nn.ReLU(inplace=True)]
        self.model_block += [nn.AdaptiveAvgPool2d(ndf*16)]
        self.fc = nn.Linear(ndf*16, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.model_block += [self.fc]

        # create model
        self.model_pre = nn.Sequential(*self.model_pre)
        self.model_block = nn.Sequential(*self.model_block)

        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, label=None):
        # if label is None:
            # print("copy x")
            # label = x.clone().cuda(0)
        assert label is not None, "plz give me label let me train discriminator"
        # print("label shape", label.shape)
        # print("label shape = ", label.shape)
        # print("proj_x shape = ", proj_x.shape)
        # pdb.set_trace()

        x = self.model_pre(x)
        proj_x = self.proj_conv(x)
        # print("proj shape", proj_x.shape)
        output = self.model_block(x)
        output += torch.sum(proj_x*label)

        return output
