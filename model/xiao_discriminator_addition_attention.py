import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import FirstResBlock_2018_SN
from .networks import ResBlock_2018_SN
from .networks import Self_Attn

from .networks import SpectralNorm

class XiaoAttentionDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(XiaoAttentionDiscriminator, self).__init__()

        self.model_pre = []
        # channe = 64
        self.model_pre += [FirstResBlock_2018_SN(num_classes, ndf, downsample=False, use_BN=False)]
        self.model_pre += [ResBlock_2018_SN(ndf, ndf, downsample=True, use_BN=False)]
        # channe = 128
        self.model_pre += [ResBlock_2018_SN(ndf, ndf * 2, downsample=False, use_BN=False)]
        self.model_pre += [ResBlock_2018_SN(ndf * 2, ndf * 2, downsample=True, use_BN=False)]

        # channel = 128
        self.model_pre += [ResBlock_2018_SN(ndf * 2, ndf * 4, downsample=False, use_BN=False)]
        # use cGANs with projection
        # channel = 256
        self.model_pre += [ResBlock_2018_SN(ndf * 4, ndf * 4, downsample=True, use_BN=False)]

        # self.model_pre += [SpectralNorm(nn.Conv2d(ndf * 4, ndf * 4, kernel_size=4, stride=2, padding=1))]
        # self.proj_conv = SpectralNorm(nn.Conv2d(ndf * 4, num_classes, kernel_size=3, stride=1, padding=1))

        # self.proj_block = []
        # channel = 512
        # self.model_pre += [ResBlock_2018_SN(ndf * 4, ndf * 8, downsample=False, use_BN=False)]
        # self.proj_block += [SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1))]
        # self.proj_block += [SpectralNorm(nn.Conv2d(ndf * 8, ndf * 16, kernel_size=3, stride=1, padding=1))]

        # channel = 1024
        # self.proj_block += [ResBlock_2018_SN(ndf * 8, ndf * 16, downsample=True, use_BN=False)]

        # self.proj_block += [ResBlock_2018_SN(num_classes, 1, downsample=True, use_BN=False)]
        # self.proj_block += [nn.ReLU()]
        # self.proj_block += [ResBlock_2018_SN(ndf*2, ndf*4, downsample=False, use_BN=False)]
        # self.proj_block += [ResBlock_2018_SN(ndf*4, 1, downsample=False, use_BN=False)]

        # self.model_block = []
        # channel = 512
        # self.model_block += [ResBlock_2018_SN(ndf * 8, ndf * 16, downsample=True, use_BN=False)]
        # channel = 1024
        # self.model_block += [ResBlock_2018_SN(ndf * 4, ndf * 2, downsample=True, use_BN=False)]

        # self.model_block += [ResBlock_2018_SN(ndf * 2, num_classes, downsample=True, use_BN=False)]

        self.model_block = []
        # channel = 512
        self.model_block += [ResBlock_2018_SN(ndf * 4, ndf * 8, downsample=False, use_BN=False)]
        # channel = 1024
        self.model_block += [ResBlock_2018_SN(ndf * 8, num_classes, downsample=True, use_BN=False)]

        # create attention model
        self.model_attn = []
        self.model_attn += [SpectralNorm(nn.Conv2d(ndf * 4, num_classes, 4, 2, 1))]
        self. model_attn += [nn.LeakyReLU(0.1)]
        self.attn1 = Self_Attn(num_classes, 'relu')
        self.model_attn += [self.attn1]


        # create classifier model
        self.model_classifier = [ResBlock_2018_SN(num_classes, 1, downsample=False, use_BN=False)]


        # create sequential model
        self.model_pre = nn.Sequential(*self.model_pre)
        self.model_block = nn.Sequential(*self.model_block)
        # self.proj_block = nn.Sequential(*self.proj_block)
        self.model_attn = nn.Sequential(*self.model_attn)
        self.model_classifier = nn.Sequential(*self.model_classifier)


    def forward(self, x, label=None, model_attn=None):
        x = self.model_pre(x)
        # print("x shape", x.shape)
        attn_out = self.model_attn(x)
        x = self.model_block(x)

        # print("out shape", out.shape)
        # attn_out = self.model_attn(x)
        # proj_x = self.proj_block(x)
        # print("proj_x shape", proj_x.shape)
        # model attn input: channel 1024, output: channel num_classes
        # print("attn_out shape", attn_out.shape)

        # use attention
        out = self.model_classifier(attn_out * x)

        return out, attn_out
