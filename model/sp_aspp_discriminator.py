import torch.nn as nn
import torch.nn.functional as F
from .networks import SpectralNorm


class SP_ASPP_FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(SP_ASPP_FCDiscriminator, self).__init__()

		self.conv1 = SpectralNorm(nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1))
		self.conv2 = SpectralNorm(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1))
		self.conv3 = SpectralNorm(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1))
		self.conv4 = SpectralNorm(nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1))
		# self.conv5 = nn.Conv2d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=1)
		# self.classifier = nn.Conv2d(ndf*16, 1, kernel_size=4, stride=2, padding=1)

		# self.classifier = SpectralNorm(nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1))
		pyramids = [4, 8, 12, 16]
		# self.aspp = _ASPPModule(ndf*8, ndf*8, pyramids)

		# self.classifier = SpectralNorm(nn.Conv2d(ndf*8, 1, kernel_size=1, stride=1, padding=0))
		self.classifier = _ASPPModule(ndf*8, 1, pyramids)


		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		# self.activation = nn.PReLU()
		self.activation = self.leaky_relu

	#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x, label=None):
		x = self.conv1(x)
		x = self.activation(x)
		x = self.conv2(x)
		x = self.activation(x)
		x = self.conv3(x)
		x = self.activation(x)
		x = self.conv4(x)
		# print("conv4 x shape =", x.shape)
		x = self.activation(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x)

		return x, None

class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling
    copy from https://github.com/kazuto1011/deeplab-pytorch/blob/3da05542c0d04502a7a23be13a8e4ec539c4670d/libs/models/deeplabv2.py"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(_ASPPModule, self).__init__()
        self.stages = nn.Module()
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                ),
            )

        for m in self.stages.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = 0
        for stage in self.stages.children():
            h += stage(x)
        return h
