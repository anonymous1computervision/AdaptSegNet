import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import FirstResBlock_2018_SN
from .networks import ResBlock_2018_SN
from .networks import Self_Attn

# from .networks import SpectralNorm

class XiaoAttentionDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(XiaoAttentionDiscriminator, self).__init__()

        self.c_block = []
        # channe = 64
        self.c_block += [SpectralNorm(nn.Conv2d(num_classes, ndf, 4, 2, 1))]
        self.c_block += [nn.LeakyReLU(0.1)]
        self.c_block += [SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1))]
        self.c_block += [nn.LeakyReLU(0.1)]
        self.c_block += [SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1))]
        self.c_block += [nn.LeakyReLU(0.1)]
        # self attention 1
        self.c_block += [Self_Attn(ndf*4, activation='relu')]
        # self.c_block += [SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1))]
        # self.c_block += [nn.LeakyReLU(0.1)]
        # # self attention 2
        # self.c_block += [Self_Attn(ndf*8, activation='relu')]
        # self.c_block += [SpectralNorm(nn.Conv2d(ndf * 8, 1, 4, 1, 1))]
        self.c_block += [SpectralNorm(nn.Conv2d(ndf * 4, 1, 4, 1, 1))]

        # create sequential model
        self.c_block = nn.Sequential(*self.c_block)


    def forward(self, x, label=None):
        out = self.c_block(x)

        return out
