import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
import torchvision.models as models

from .networks import FirstResBlock_2018_SN
from .networks import ResBlock_2018_SN
from .networks import Self_Attn
from .networks import SpectralNorm
from util import weights_init

class XiaoPretrainAttentionDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(XiaoPretrainAttentionDiscriminator, self).__init__()

        # channel 19 to 3
        # self.model_pre = []
        # self.model_pre += [FirstResBlock_2018_SN(num_classes, 64, downsample=False, use_BN=False)]

        self.model_pre = []
        self.model_pre += [SpectralNorm(nn.Conv2d(num_classes, ndf, 4, 2, 1))]
        self.model_pre += [nn.LeakyReLU(0.1)]
        self.model_pre += [SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1))]
        self.model_pre += [nn.LeakyReLU(0.1)]
        self.model_pre += [SpectralNorm(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1))]
        self.model_pre += [nn.LeakyReLU(0.1)]
        self.model_pre += [SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1))]
        self.model_pre += [nn.LeakyReLU(0.1)]

        # self.model_pre += [ResBlock_2018_SN(ndf, ndf, downsample=True, use_BN=False)]
        # # channe = 128
        # self.model_pre += [ResBlock_2018_SN(ndf, ndf * 2, downsample=False, use_BN=False)]
        # self.model_pre += [ResBlock_2018_SN(ndf * 2, ndf * 2, downsample=True, use_BN=False)]
        #
        # # channel = 128
        # self.model_pre += [ResBlock_2018_SN(ndf * 2, ndf * 4, downsample=False, use_BN=False)]
        # # use cGANs with projection
        # # channel = 256
        # self.model_pre += [ResBlock_2018_SN(ndf * 4, ndf * 4, downsample=True, use_BN=False)]

        self.proj_conv = SpectralNorm(nn.Conv2d(ndf * 4, num_classes, kernel_size=4, stride=2, padding=1))
        # self.proj_block = []
        # self.proj_block += [ResBlock_2018_SN(num_classes, 1, downsample=True, use_BN=False)]
        # self.proj_block += [nn.ReLU()]
        # self.proj_block += [ResBlock_2018_SN(ndf*2, ndf*4, downsample=False, use_BN=False)]
        # self.proj_block += [ResBlock_2018_SN(ndf*4, 1, downsample=False, use_BN=False)]

        self.model_block = []
        # self.model_block += [SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1))]
        # self.model_block += [nn.LeakyReLU(0.1)]
        self.model_block += [SpectralNorm(nn.Conv2d(ndf * 4, num_classes, 4, 2, 1))]
        # self.model_block += [nn.ReLU()]
        self.model_block += [nn.LeakyReLU(0.1)]
        # channel = 512
        # self.model_block += [ResBlock_2018_SN(restnet_out_c, restnet_out_c*2, downsample=False, use_BN=False)]
        # channel = 1024
        # self.model_block += [ResBlock_2018_SN(restnet_out_c*2, num_classes, downsample=True, use_BN=False)]


        # use pretrianed model
        # self.resnet18 = models.resnet18(pretrained=True)
        self.densenet121 = models.densenet121(pretrained=True)

        # remove first 3 to n_channel and last fc layer
        # modules = list(self.resnet18.children())[:-1]
        modules = list(self.densenet121.children())[:-1]
        # for n_layer, p in enumerate(modules):
        #     print("layer =", n_layer)
        #     print("p =", p)


        # restnet_out_c = 512
        densenet_out_c = 1024

        # create attention model
        model_attn = []
        model_attn += [SpectralNorm(nn.Conv2d(densenet_out_c, num_classes, 3, 1, 1))]
        model_attn += [nn.LeakyReLU(0.1)]
        self.attn1 = Self_Attn(num_classes, 'relu')
        # self.attn2 = Self_Attn(ndf*8, 'relu')

        model_attn += [self.attn1]
        # model_attn += [SpectralNorm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1))]
        # model_attn += [nn.LeakyReLU(0.1)]
        # model_attn += [self.attn2]

        # create classifier model
        # self.model_classifier = [ResBlock_2018_SN(num_classes, 1, downsample=False, use_BN=False)]
        self.model_classifier = [SpectralNorm(nn.Conv2d(num_classes, 1, 4, 2, 1))]


        # create sequential model
        self.model_pre = nn.Sequential(*self.model_pre)
        self.model_block = nn.Sequential(*self.model_block)
        # self.proj_block = nn.Sequential(*self.proj_block)
        self.model_attn = nn.Sequential(*model_attn)
        self.model_classifier = nn.Sequential(*self.model_classifier)

        # use weight init
        self.model_pre.apply(weights_init("xavier"))
        self.model_block.apply(weights_init("xavier"))
        self.model_attn.apply(weights_init("xavier"))
        self.model_classifier.apply(weights_init("xavier"))

        # self.resnet18 = nn.Sequential(*modules)
        # for p in self.resnet18.parameters():
        #     p.requires_grad = False

        self.densenet121 = nn.Sequential(*modules)
        for p in self.densenet121.parameters():
            p.requires_grad = False



    def forward(self, x, label=None):

        x = self.model_pre(x)
        out = self.model_block(x)
        # print("model pre x shape", x.shape)
        # print("out smodel_block hape", out.shape)


        #####################
        #  conditional GAN  #
        #####################
        with torch.no_grad():
            # cond_y = self.resnet18(label)
            cond_y = self.densenet121(label)
            # print("cond_y shape =", cond_y.shape)

        attn_out = self.model_attn(cond_y)
        # print("attn_out shape", attn_out.shape)

        proj = self.proj_conv(x)
        # print("proj shape", proj.shape)

        attn_out = proj * attn_out

        # x = self.model_pre(x)
        # print("model pre x shape", x.shape)


        # use addition attention
        out = out + attn_out
        out = self.model_classifier(out)

        return out, attn_out
