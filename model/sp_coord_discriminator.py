import torch
import torch.nn as nn
import torch.nn.functional as F
# from .networks import SpectralNorm
# from .spectral import spectral_norm
from torch.nn.utils import spectral_norm
import numpy as np

class SP_Coord_FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(SP_Coord_FCDiscriminator, self).__init__()

		self.conv1 = spectral_norm(nn.Conv2d(num_classes+2, ndf, kernel_size=4, stride=2, padding=1))
		self.conv2 = spectral_norm(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1))
		self.conv3 = spectral_norm(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1))
		self.conv4 = spectral_norm(nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1))
		# self.conv5 = nn.Conv2d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=1)
		# self.classifier = nn.Conv2d(ndf*16, 1, kernel_size=4, stride=2, padding=1)
		self.classifier = spectral_norm(nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1))

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		# self.activation = nn.PReLU()
		self.activation = self.leaky_relu

	#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()
		# Init weights
	def _add_xy_index_channels(self, img):
		h_s = img.shape[-2]
		w_s = img.shape[-1]
		c_y_index = np.expand_dims(np.tile(np.arange(h_s), w_s).reshape(w_s, h_s).T, 2) / (h_s - 1)
		c_x_index = np.tile(np.arange(w_s), h_s).T.reshape(h_s, w_s, 1) / (w_s - 1)
		c_y_index = torch.tensor(c_y_index,
									 dtype=torch.float64,
									 device=torch.device('cuda:0')).float().cuda().view(1, 1, h_s, w_s)
		c_x_index = torch.tensor(c_x_index,
									 dtype=torch.float64,
									 device=torch.device('cuda:0')).float().cuda().view(1, 1, h_s, w_s)

		# img_extended = np.concatenate([img, c_y_index, c_x_index], 2).copy()
		img_extended = torch.cat([img, c_y_index, c_x_index], 1)

		return img_extended

	def forward(self, x, label=None):
		x = self._add_xy_index_channels(x)

		x = self.conv1(x)
		x = self.activation(x)
		x = self.conv2(x)
		x = self.activation(x)
		x = self.conv3(x)
		x = self.activation(x)
		x = self.conv4(x)
		x = self.activation(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x)

		return x, None
