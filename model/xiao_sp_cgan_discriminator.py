import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import FirstResBlock_2018_SN
from .networks import ResBlock_2018_SN
from .networks import Self_Attn
from model.networks import StyleEncoder, MLP
from .networks import SpectralNorm
from .networks import AdaptiveInstanceNorm2d
# import pdb

class XiaoCganDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(XiaoCganDiscriminator, self).__init__()

        # todo:print gamma
        self.gamma = nn.Parameter(torch.zeros(1))


        # ==================== #
        #    model pre         #
        # ==================== #
        self.model_pre = []
        # # channe = 64
        self.model_pre.append(SpectralNorm(nn.Conv2d(num_classes, ndf, 4, 2, 1)))
        self.model_pre += [nn.LeakyReLU(0.2)]
        # # channe = 128
        self.model_pre.append(SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)))
        self.model_pre += [nn.LeakyReLU(0.2)]
        # # channe = 256
        self.model_pre.append(SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)))
        self.model_pre += [nn.LeakyReLU(0.2)]
        # self.model_pre = []
        # channe = 64
        # self.model_pre += [FirstResBlock_2018_SN(num_classes, ndf, downsample=False, use_BN=False)]
        # self.model_pre += [ResBlock_2018_SN(ndf, ndf, downsample=True, use_BN=False)]
        # channe = 128
        # self.model_pre += [ResBlock_2018_SN(ndf, ndf * 2, downsample=False, use_BN=False)]
        # self.model_pre += [ResBlock_2018_SN(ndf * 2, ndf * 2, downsample=True, use_BN=False)]

        # channel = 128
        # self.model_pre += [ResBlock_2018_SN(ndf * 2, ndf * 4, downsample=False, use_BN=False)]
        # use cGANs with projection
        # channel = 256
        # self.model_pre += [ResBlock_2018_SN(ndf * 4, ndf * 4, downsample=True, use_BN=False)]


        # ==================== #
        #     self-attention   #
        # ==================== #
        # self.c_block = []
        # self.c_block += [SpectralNorm(nn.Conv2d(3, ndf, 4, 2, 1))]
        # self.c_block += [nn.LeakyReLU(0.2)]
        # self.c_block += [SpectralNorm(nn.Conv2d(ndf, ndf*2, 4, 2, 1))]
        # self.c_block += [nn.LeakyReLU(0.2)]
        # self.c_block += [SpectralNorm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1))]
        # self.c_block += [nn.LeakyReLU(0.2)]
        # self.c_block += [SpectralNorm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1))]
        # self.c_block += [nn.LeakyReLU(0.2)]
        # self.c_block += [SpectralNorm(nn.Conv2d(ndf, ndf, 4, 2, 1))]
        # self.c_block += [nn.LeakyReLU(0.1)]

        # self.c_block += [ResBlock_2018_SN(ndf, ndf * 2, downsample=True, use_BN=False)]
        # self.c_block += [ResBlock_2018_SN(ndf * 2, ndf * 2, downsample=True, use_BN=False)]
        # self.c_block += [ResBlock_2018_SN(ndf * 2, ndf * 4, downsample=False, use_BN=False)]
        # self.c_block += [ResBlock_2018_SN(ndf * 4, ndf * 4, downsample=False, use_BN=False)]
        # self.c_block += [Self_Attn(ndf * 8, 'relu')]
        # self.c_block += [ResBlock_2018_SN(ndf * 8, num_classes, downsample=False, use_BN=False)]


        # self.c_block.append(nn.LeakyReLU(0.1))
        # self.c_block.append(SpectralNorm(nn.Conv2d(ndf, ndf * 2, 3, 1, 1)))
        # self.c_block.append(nn.LeakyReLU(0.1))
        # self.c_block.append(SpectralNorm(nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1)))
        # self.c_block.append(SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1)))
        # self.c_block.append(nn.LeakyReLU(0.1))

        # self.c_block = nn.Sequential(*self.c_block)


        # use cGANs with projection
        # ==================== #
        #   proj conv          #
        # ==================== #
        self.proj_conv = []
        # self.proj_conv += [nn.ReLU()]
        self.proj_conv += [SpectralNorm(nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1, padding=1))]
        self.proj_conv += [nn.LeakyReLU(0.2)]
        # self.proj_conv += [ResBlock_2018_SN(ndf * 4, ndf * 4, downsample=False, use_BN=False)]
        # self.proj_conv += [nn.ReLU()]
        # use self attention too
        # self.proj_attn = Self_Attn(ndf * 4, 'relu')
        # self.proj_conv += [self.proj_attn]
        # self.proj_conv += [nn.LeakyReLU(0.2)]
        # self.proj_attn = Self_Attn(ndf * 4, 'relu')
        # self.proj_conv += [self.proj_attn]
        self.proj_conv += [nn.Conv2d(ndf * 4, 1, kernel_size=3, stride=1, padding=1)]
        self.proj_conv += [nn.ReLU()]
        # self.proj_conv += [nn.LeakyReLU(0.2)]
        # todo:check tanh
        # self.proj_conv += [nn.Tanh()]


        # ==================== #
        #   model_block        #
        # ==================== #
        # self.model_block = []
        # channel = 512
        # self.model_block.append(SpectralNorm(nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1)))


        # use some trick
        # self.model_block += [nn.ReLU(inplace=True)]
        # self.model_block += [nn.AdaptiveAvgPool2d(ndf * 8)]

        self.model_block = []
        # channel = 512
        self.model_block += [SpectralNorm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1))]
        # self.model_block += [nn.LeakyReLU(0.2)]
        # self.model_block += [ResBlock_2018_SN(ndf * 4, ndf * 8, downsample=True, use_BN=False)]
        # channel = 1024
        # self.model_block += [ResBlock_2018_SN(ndf * 8, num_classes, downsample=False, use_BN=False)]
        self.model_block += [nn.ReLU(inplace=True)]
        self.model_block += [nn.AdaptiveAvgPool2d(ndf * 8)]
        # ==================== #
        #          fc          #
        # ==================== #
        self.fc = nn.Linear(ndf * 8, 1)
        # todo:this initial will check
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        # nn.init.xavier_uniform_(self.fc.weight.data)

        self.model_block += [self.fc]


        # self.c_block_2 = []
        # self.c_block_2.append(SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)))
        # self.c_block_2.append(nn.LeakyReLU(0.1))
        # self.c_block_2 = nn.Sequential(*self.c_block_2)

        # self.attn1 = Self_Attn(num_classes, 'relu')
        # self.attn2 = Self_Attn(512, 'relu')

        # ==================== #
        #     adaptive IN      #
        # ==================== #
        # STYLE_DIM = 8
        # MLP_DIM = 256
        # self.enc_style = StyleEncoder(4, input_dim=3, dim=64, style_dim=STYLE_DIM, norm='none', activ="relu",
        #                               pad_type="reflect")
        # # MLP to generate AdaIN parameters
        # self.mlp = MLP(STYLE_DIM, self.get_num_adain_params(self.c_block), MLP_DIM, 3, norm='none', activ="relu")

        # ==================== #
        #     model_classifier #
        # ==================== #
        # self.model_classifier = [ResBlock_2018_SN(num_classes, 1, downsample=False, use_BN=False)]
        # create model
        self.model_pre = nn.Sequential(*self.model_pre)
        self.model_block = nn.Sequential(*self.model_block)
        self.proj_conv = nn.Sequential(*self.proj_conv)
        # self.model_classifier = nn.Sequential(*self.model_classifier)

    def forward(self, x, label=None):
        # if label is None:
        # print("copy x")
        # label = x.clone().cuda(0)
        assert label is not None, "plz give me label let me train discriminator"
        # print("label shape", label.shape)
        # print("label shape = ", label.shape)

        # inter_mini = nn.Upsample(size=(int(label.shape[-2] / 8), int(label.shape[-1] / 8)), align_corners=False,
        #                          mode='bilinear')
        # style_code = self.enc_style(inter_mini(label))
        # self.adain(style_code)

        # print("x shape =", x.shape)
        # conditional y
        # print("y shape =", label.shape)
        # c_out = self.c_block(c_edge)
        # attn1 = self.attn1(y)
        # print("y1 shape =", y.shape)
        # y = self.c_block_2(y)
        # y = self.attn2(y)
        # print("y2 shape =", y.shape)
        x = self.model_pre(x)
        # print("model pre x shape = ", x.shape)

        proj_x = self.proj_conv(x)
        # print("proj_x shape = ", proj_x.shape)

        output = self.model_block(x)
        # print("model_block shape = ", output.shape)

        # print("proj_x shape =", proj_x.shape)
        # print("c_out shape", c_out.shape)
        # print("output shape", output.shape)
        # output += torch.sum(proj_x*label)
        # todo: check gamma can robust model?
        # output = (1-self.gamma)*output + self.gamma*torch.sum(proj_x*label)
        output += self.gamma*torch.sum(proj_x*label)
        # output += torch.sum(proj_x*label)

        # output = self.model_block(x)
        # output += proj_x
        # output += torch.sum(proj_x*label)
        # output += torch.mean(proj_x * label)
        # normalize = 8 / 256
        # output += torch.sum(proj_x * y) * normalize

        return output, proj_x


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
