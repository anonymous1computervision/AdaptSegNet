import torch
import torch.nn as nn
import torch.nn.functional as F
# from .networks import SpectralNorm
# from .spectral import spectral_norm
from torch.nn.utils import spectral_norm
import scipy.io as sio
import numpy as np

from util.util import weights_init

class SP_CGAN_FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(SP_CGAN_FCDiscriminator, self).__init__()
		print("in sp cgan")
		# self.gamma = nn.Parameter(torch.zeros(1))
		# self.gamma = nn.Parameter(torch.zeros(num_classes))
		self.foreground_map = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
		n = len(self.foreground_map)
		self.gamma = nn.Parameter(torch.zeros(n))
		self.conv1 = spectral_norm(nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1))
		# self.conv1 = spectral_norm(nn.Conv2d(num_classes+2, ndf, kernel_size=4, stride=2, padding=1))
		self.conv2 = spectral_norm(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1))
		self.conv3 = spectral_norm(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1))
		self.conv4 = spectral_norm(nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1))
		# self.gamma = nn.Parameter(torch.zeros(1))

		# self.classifier = spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1))

		self.fc = [spectral_norm(nn.Linear(ndf * 8, 1))]
		self.fc = nn.Sequential(*self.fc)
		weights_init(self.fc)

		self.spatial_matrix = self.get_spatial_matrix()
		C, H, W = self.spatial_matrix.shape
		self.spatial_matrix = self.spatial_matrix.view(1, C, H, W)

		# print("spatial_matrix shape =", self.spatial_matrix.shape)
		# num_classes = 11
		# proj part

		self.proj_conv = []
		# self.proj_conv += [spectral_norm(nn.Conv2d(ndf * 4, num_classes, kernel_size=3, stride=1, padding=1))]
		# self.proj_conv += [spectral_norm(nn.Conv2d(ndf * 4, n, kernel_size=3, stride=1, padding=1))]
		self.proj_conv += [spectral_norm(nn.Conv2d(ndf, n, kernel_size=3, stride=1, padding=1))]

		self.proj_conv = nn.Sequential(*self.proj_conv)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.activation = self.leaky_relu

	# self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
	# self.sigmoid = nn.Sigmoid()
	# Init weights

	def get_spatial_matrix(self, path="./model/prior_array.mat"):
		# print("get_spatial_matrix")
		sprior = sio.loadmat(path)
		sprior = sprior["prior_array"]
		# print("sprior shaps =", sprior.shape)
		sprior = sprior[self.foreground_map]
		# print(" new sprior shaps =", sprior.shape)
		tensor_sprior = torch.tensor(sprior,
									 dtype=torch.float64,
									 device=torch.device('cuda:0')).float().cuda()
		# tensor_sprior = 1 + tensor_sprior
		# tensor_sprior = tensor_sprior.double()
		return tensor_sprior

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

	def forward(self, input, label=None):
		# assert label is not None
		assert label is None

		# origin label
		# 1=foreground 0=background
		# mask = 1 - nn.Sigmoid()(label)
		# mask = 1 - label
		# mask = label
		# threshold = 0.5
		# threshold = 0.7
		# label[label >= threshold] = 1
		# label[label < threshold] = 0

		# x = torch.cat((input, mask), dim=1)
		# x = self._add_xy_index_channels(x)
		x = self.conv1(input)
		x = self.activation(x)
		proj = self.proj_conv(x)
		proj_shape = proj.shape
		x = self.conv2(x)
		x = self.activation(x)
		x = self.conv3(x)
		x = self.activation(x)
		# print("x shape =", x.shape)

		# =================
		# project part
		# =================
		# self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')

		# print("proj shape =", proj_shape)
		# print("proj type =", proj.type())

		interp = nn.Upsample(size=proj_shape[-2:], align_corners=False, mode='bilinear')
		# print("self.spatial_matrix type =", self.spatial_matrix.type())
		spatial_matrix = interp(self.spatial_matrix)
		input_foreground = input[:, self.foreground_map]
		input_foreground_resize = interp(input_foreground).detach()

		# print("spatial_matrix type =", spatial_matrix.type())
		# print("spatial_matrix shape =", spatial_matrix.shape)
		# print("input_foreground_resize shape =", input_foreground_resize.shape)

		# spatial_matrix = spatial_matrix.double()
		# proj = proj.float()
		spatial_info = (1 + spatial_matrix) * input_foreground_resize
		proj = proj * spatial_info
		# proj = proj * spatial_matrix
		# print("proj shape =", proj.shape)
		# proj = torch.sum(proj, dim=(1, 2, 3))
		proj = torch.sum(proj, dim=(2, 3))
		# proj = proj
		# print("proj shape =", proj.shape)
		# print("proj shape =", proj.shape)

		# =================
		# block
		# =================
		x = self.conv4(x)
		x = self.activation(x)

		x = torch.sum(x, dim=(2, 3))
		x = self.fc(x)
		# x = self.classifier(x)
		# x = x + proj
		# print(" x.shape=",  x.shape)

		# print(" self.gamma=",  self.gamma.shape)
		# print("proj shape=", proj.shape)
		# x = x + self.gamma * proj
		x = x + proj
		# print(" x.spe=",  x.shape)

		# x += proj

		# print(" x.shape=",  x.shape)

		# x = self.up_sample(x)
		# x = self.sigmoid(x)

		return x, None
