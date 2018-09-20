import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import Gated_conv
# from .networks import SpectralNorm
from torch.nn.utils import spectral_norm

class Gated_Discriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Gated_Discriminator, self).__init__()
		# self.conv1 = Gated_conv(num_classes, ndf, kernel_size=5, stride=1, padding=2)
		# self.conv2 = Gated_conv(ndf, ndf * 2, kernel_size=5, stride=2, padding=2)
		# self.conv3 = Gated_conv(ndf * 2, ndf * 4, kernel_size=5, stride=2, padding=2)
		# self.conv4 = Gated_conv(ndf * 4, ndf * 4, kernel_size=5, stride=2, padding=2)
		# self.conv5 = Gated_conv(ndf * 4, ndf * 4, kernel_size=5, stride=2, padding=2)
		# self.conv6 = Gated_conv(ndf * 4, ndf * 4, kernel_size=5, stride=2, padding=2)
		self.conv1 = Gated_conv(num_classes, ndf, kernel_size=5, stride=2, padding=2)
		self.conv2 = Gated_conv(ndf, ndf * 2, kernel_size=5, stride=2, padding=2)
		self.conv3 = Gated_conv(ndf * 2, ndf * 2, kernel_size=5, stride=2, padding=2)
		self.conv4 = Gated_conv(ndf * 2, ndf * 4, kernel_size=5, stride=2, padding=2)
		self.conv5 = Gated_conv(ndf * 4, ndf * 4, kernel_size=5, stride=2, padding=2)
		self.conv6 = Gated_conv(ndf * 4, ndf * 8, kernel_size=5, stride=2, padding=2)
		# self.classifier = spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1))
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.activation = self.leaky_relu


	def forward(self, x, label=None):
		assert label is not None
		# origin label
		# 1=foreground 0=background
		# mask = 1 - nn.Sigmoid()(label)
		mask = 1 - label
		x = torch.cat((x, mask), dim=1)
		x = self.conv1(x)
		# x = self.activation(x)
		x = self.conv2(x)
		# x = self.activation(x)
		x = self.conv3(x)
		# x = self.activation(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)

		# print("x out =", x)
		# x = self.activation(x)
		# x = self.classifier(x)
		# x = self.conv5(x)
		# x = self.activation(x)
		# x = self.conv6(x)
		# x = self.activation(x)
		return x, None
