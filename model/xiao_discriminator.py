import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .networks import ResBlock_2018
from .networks import FirstResBlock_2018_SN
from .networks import ResBlock_2018_SN
from .networks import SpectralNorm

class XiaoDiscriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, num_classes, ndf = 64):
        super(XiaoDiscriminator, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1))
        self.conv3 = SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1))
        self.conv4 = SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1))
        # self.conv5 = nn.Conv2d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=1)
        # self.classifier = nn.Conv2d(ndf*16, 1, kernel_size=4, stride=2, padding=1)
        self.classifier = SpectralNorm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.activation = nn.PReLU()
        self.activation = self.leaky_relu

    def forward(self, x, label=None):
        # if label is None:
            # print("copy x")
            # label = x.clone().cuda(0)
        # assert label is not None, "plz give me label let me train discriminator"
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.classifier(x)

        return x

