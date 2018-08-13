import torch
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
    copy from https://github.com/gengyanlei/deeplab_v3/blob/master/model.py"""

    class ASPP(nn.Module):
        def __init__(self, in_channel=512, depth=256):
            super().__init__()
            # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
            self.mean = nn.AdaptiveAvgPool2d((1, 1))
            self.conv = nn.Conv2d(in_channel, depth, 1, 1)
            # k=1 s=1 no pad
            self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
            self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
            self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
            self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

            self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
            

        def forward(self, x):
            size = x.shape[2:]

            image_features = self.mean(x)
            image_features = self.conv(image_features)
            image_features = F.upsample(image_features, size=size, mode='bilinear', align_corners=True)

            atrous_block1 = self.atrous_block1(x)

            atrous_block6 = self.atrous_block6(x)

            atrous_block12 = self.atrous_block12(x)

            atrous_block18 = self.atrous_block18(x)

            net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                                  atrous_block12, atrous_block18], dim=1))
            return net