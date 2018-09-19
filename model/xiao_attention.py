import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import FirstResBlock_2018_SN
from .networks import ResBlock_2018_SN
from .networks import Self_Attn

# from .networks import SpectralNorm

class XiaoAttention(nn.Module):

    def __init__(self, num_classes=19, ndf=1024):
        super(XiaoAttention, self).__init__()

        # create attention model
        model_attn = []
        model_attn += [SpectralNorm(nn.Conv2d(ndf, num_classes, 4, 2, 1))]

        model_attn += [nn.LeakyReLU(0.1)]
        self.attn1 = Self_Attn(num_classes, 'relu')
        model_attn += [self.attn1]

        self.model_attn = nn.Sequential(*model_attn)


    def forward(self, x):
        # x = self.model_pre(x)
        # # print("x shape", x.shape)
        #
        # out = self.model_block(x)
        # # print("out shape", out.shape)
        #
        attn_out = self.model_attn(x)
        # print("attn_out shape", attn_out.shape)
        #
        # # use attention
        # out = out * attn_out
        # out = self.model_classifier(out)

        return attn_out
