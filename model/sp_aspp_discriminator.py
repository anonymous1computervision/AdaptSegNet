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
		# pyramids = [4, 8, 12, 16]
		# self.aspp = _ASPPModule(ndf*8, ndf*8, pyramids)

		# self.classifier = SpectralNorm(nn.Conv2d(ndf*8, 1, kernel_size=1, stride=1, padding=0))
		# ASPP
		rates = [1, 6, 12, 18]
		self.aspp1 = ASPP_module(ndf*8, 256, rate=rates[0])
		self.aspp2 = ASPP_module(ndf*8, 256, rate=rates[1])
		self.aspp3 = ASPP_module(ndf*8, 256, rate=rates[2])
		self.aspp3 = ASPP_module(ndf*8, 256, rate=rates[2])

		# self.aspp4 = ASPP_module(ndf*8, 256, rate=rates[3])
		# self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
		#                                      nn.Conv2d(2048, 256, 1, stride=1, bias=False),
		#                                      # todo: because batch size = 1 so disable
		#                                      # nn.BatchNorm2d(256),
		#                                      nn.ReLU())
		
		self.conv_depth = SpectralNorm(nn.Conv2d(256*4, 256, kernel_size=1, stride=1, padding=0))

		self.classifier = SpectralNorm(nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1))
		

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
		x1 = self.aspp1(x)
		x2 = self.aspp2(x)
		x3 = self.aspp3(x)
		x4 = self.aspp4(x)
		
		# x5 = self.global_avg_pool(x)
		# x5 = F.upsample(x5, size=x3.size()[2:], mode='bilinear', align_corners=True)
		
		# x = torch.cat((x1, x2, x3, x5), dim=1)
		x = torch.cat((x1, x2, x3, x4), dim=1)

		x = self.activation(x)
		# print("in here x shape", x.shape)
		x = self.conv_depth(x)
		x = self.activation(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x)

		return x, None

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()