import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .networks import ResBlock_2018
from .networks import FirstResBlock_2018_SN
from .networks import ResBlock_2018_SN
from .networks import SpectralNorm
from .networks import SpectralNorm
from .networks import Self_Attn

class XiaoDiscriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, conv_dim=32, num_classes=19):
        super(XiaoDiscriminator, self).__init__()
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(num_classes, conv_dim, kernel_size=4, stride=2, padding=1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2


        layer4 = []
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
        layer4.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)


        self.attn1 = Self_Attn(128, 'relu')
        self.attn2 = Self_Attn(256, 'relu')

        last.append(nn.Conv2d(256, 1, 4))
        self.last = nn.Sequential(*last)


        # self.attn1 = Self_Attn(256, 'relu')
        # self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x, label=None):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)

        out, p2 = self.attn2(out)
        out = self.last(out)

        return out.squeeze()