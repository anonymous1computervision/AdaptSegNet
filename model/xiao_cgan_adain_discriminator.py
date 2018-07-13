import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import FirstResBlock_2018_SN
from .networks import ResBlock_2018_SN
from .networks import Self_Attn
from model.networks import StyleEncoder, MLP
from .networks import SpectralNorm
from .networks import AdaptiveInstanceNorm2d

class XiaoCganDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(XiaoCganDiscriminator, self).__init__()



        # ==================== #
        #    model pre         #
        # ==================== #
        self.model_pre = []
        # channe = 64
        self.model_pre.append(SpectralNorm(nn.Conv2d(num_classes, ndf, 4, 2, 1)))
        self.model_pre += [nn.LeakyReLU(0.1)]
        # channe = 128
        self.model_pre.append(SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)))
        self.model_pre += [nn.LeakyReLU(0.1)]
        # channe = 256
        self.model_pre.append(SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)))
        self.model_pre += [nn.LeakyReLU(0.1)]


        # use cGANs with projection
        # ==================== #
        #   proj conv          #
        # ==================== #
        self.proj_conv = []
        self.proj_conv += [SpectralNorm(nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1, padding=1))]
        self.proj_conv += [nn.ReLU(inplace=True)]

        # ==================== #
        #   model_block        #
        # ==================== #
        self.model_block = []
        # channel = 512
        self.model_block.append(SpectralNorm(nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1)))
        # use some trick
        # self.model_block += [nn.ReLU(inplace=True)]
        # self.model_block += [nn.AdaptiveAvgPool2d(ndf * 8)]

        # ==================== #
        #          fc          #
        # ==================== #
        # self.fc = nn.Linear(ndf * 8, 1)
        # nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        # self.model_block += [self.fc]

        # ==================== #
        #     self-attention   #
        # ==================== #
        self.c_block = []
        self.c_block.append(SpectralNorm(nn.Conv2d(3, ndf, 4, 2, 1)))
        # instanced normalized
        self.in1 = AdaptiveInstanceNorm2d(64)
        self.c_block.append(self.in1)

        # self.c_block.append(nn.LeakyReLU(0.1))
        self.c_block.append(SpectralNorm(nn.Conv2d(ndf, ndf * 2, 3, 1, 1)))
        self.c_block.append(nn.LeakyReLU(0.1))
        # self.c_block.append(SpectralNorm(nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1)))
        self.c_block.append(SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1)))
        self.c_block.append(nn.LeakyReLU(0.1))

        self.c_block = nn.Sequential(*self.c_block)

        # self.c_block_2 = []
        # self.c_block_2.append(SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)))
        # self.c_block_2.append(nn.LeakyReLU(0.1))
        # self.c_block_2 = nn.Sequential(*self.c_block_2)

        self.attn1 = Self_Attn(256, 'relu')
        # self.attn2 = Self_Attn(512, 'relu')

        # ==================== #
        #     adaptive IN      #
        # ==================== #
        STYLE_DIM = 8
        MLP_DIM = 256
        self.enc_style = StyleEncoder(4, input_dim=3, dim=64, style_dim=STYLE_DIM, norm='none', activ="relu",
                                      pad_type="reflect")
        # MLP to generate AdaIN parameters
        self.mlp = MLP(STYLE_DIM, self.get_num_adain_params(self.c_block), MLP_DIM, 3, norm='none', activ="relu")

        # create model
        self.model_pre = nn.Sequential(*self.model_pre)
        self.model_block = nn.Sequential(*self.model_block)
        self.proj_conv = nn.Sequential(*self.proj_conv)


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
        style_code = self.enc_style(label)
        self.adain(style_code)

        # conditional y
        # print("y shape =", label.shape)
        y = self.c_block(label)
        y = self.attn1(y)
        # print("y1 shape =", y.shape)
        # y = self.c_block_2(y)
        # y = self.attn2(y)
        # print("y2 shape =", y.shape)

        x = self.model_pre(x)
        proj_x = self.proj_conv(x)
        output = self.model_block(x)
        # print("proj_x shape =", proj_x.shape)
        # print("proj shape", proj_x.shape)

        # inspired by residul
        output += y*proj_x

        # output = self.model_block(x)
        # output += proj_x
        # output += torch.sum(proj_x*label)
        # output += torch.mean(proj_x * label)
        # normalize = 8 / 256
        # output += torch.sum(proj_x * y) * normalize


        return output
    def adain(self, style_code):
        adain_params = self.mlp(style_code)

        # assign the adain_params to the AdaIN layers in model
        for m in self.c_block.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            # print(m.__class__.__name__)
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                print("In discriminator I found a layer use AdaptiveInstanceNorm2d")
                num_adain_params += 2*m.num_features
        print("num_adain_params =", num_adain_params)
        return num_adain_params
