import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import Gated_conv
# from .networks import SpectralNorm
from .networks import PCBActiv

from torch.nn.utils import spectral_norm

class Partial_Discriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Partial_Discriminator, self).__init__()
		self.num_classes = num_classes
		# self.conv1 = PCBActiv(num_classes, ndf, sample='down-5', activ='leaky')
		# self.conv2 = PCBActiv(ndf, ndf * 2, sample='down-5', activ='leaky')
		# self.conv3 = PCBActiv(ndf * 2, ndf * 2, sample='down-5', activ='leaky')
		# self.conv4 = PCBActiv(ndf * 2, ndf * 4, sample='down-5', activ='leaky')
		# self.conv5 = PCBActiv(ndf * 4, ndf * 4, sample='down-5', activ='leaky')
		# self.conv6 = PCBActiv(ndf * 4, ndf * 8, sample='down-5', activ='leaky')
		self.conv1 = PCBActiv(num_classes, ndf, sample='down-4', activ='leaky')
		self.conv2 = PCBActiv(ndf, ndf * 2, sample='down-4', activ='leaky')
		self.conv3 = PCBActiv(ndf * 2, ndf * 4, sample='down-4', activ='leaky')
		self.conv4 = PCBActiv(ndf * 4, ndf * 8, sample='down-4', activ='leaky')
		self.classifier = spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1))

		# self.classifier = spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1))
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.activation = self.leaky_relu

	def forward(self, x, label=None):
		assert label is not None
		# origin label
		# 1=foreground 0=background
		# mask = 1 - nn.Sigmoid()(label)
		# add threshold
		# threshold = 0.5
		# label[label >= threshold] = 1
		# label[label < threshold] = 0
		mask = 1 - label
		batch, channel, h, w = mask.shape
		mask = mask.expand(batch, channel*self.num_classes, h, w)
		x, mask = self.conv1(x, mask)
		x, mask = self.conv2(x, mask)
		x, mask = self.conv3(x, mask)
		x, mask = self.conv4(x, mask)
		# x, mask = self.conv5(x, mask)
		# x, mask = self.conv6(x, mask)
		# x = self.activation(x)
		# x = self.conv2(x)
		# x = self.activation(x)
		# x = self.conv3(x)
		# x = self.activation(x)
		# x = self.conv4(x)
		# x = self.conv5(x)
		# x = self.conv6(x)

		# print("x out =", x)
		# x = self.activation(x)
		x = self.classifier(x)
		# x = self.conv5(x)
		# x = self.activation(x)
		# x = self.conv6(x)
		# x = self.activation(x)
		return x, None
