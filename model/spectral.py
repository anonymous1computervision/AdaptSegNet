import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

# class SpectralNorm(object):
# 	"""
# 	0917 update
# 	from https://github.com/kahartma/GAN/blob/b3fab1b0e9b3fdafc359d1aca1e4ebbf0f41d55d/eeggan/modules/layers/spectral_norm.py
# 	Implemented for PyTorch using WeightNorm implementation
# 	https://pytorch.org/docs/stable/_modules/torch/nn/utils/weight_norm.html
# 	References
# 	----------
# 	Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018).
# 	Spectral Normalization for Generative Adversarial Networks.
# 	Retrieved from http://arxiv.org/abs/1802.05957
# 	"""
# 	def __init__(self, name):
# 		self.name = name
#
# 	def compute_weight(self, module):
# 		weight = getattr(module, self.name)
# 		u = getattr(module, self.name + '_u')
#
# 		weight_size = list(weight.size())
# 		weight_tmp = weight.data.view(weight_size[0],-1)
# 		v = weight_tmp.t().matmul(u)
# 		v = v/v.norm()
# 		u = weight_tmp.matmul(v)
# 		u = u/u.norm()
# 		o = u.t().matmul(weight_tmp.matmul(v))
# 		weight_tmp = weight_tmp/o
# 		weight.data = weight_tmp.view(*weight_size)
#
# 		setattr(module, self.name + '_u', u)
# 		setattr(module, self.name, weight)
#
# 	@staticmethod
# 	def apply(module, name):
# 		fn = SpectralNorm(name)
#
# 		weight = getattr(module, name)
# 		u = torch.Tensor(weight.size(0),1)
# 		u.normal_()
#
# 		module.register_buffer(name + '_u', u)
# 		module.register_forward_pre_hook(fn)
#
# 		return fn
#
# 	def remove(self, module):
# 		del module._buffers[name + '_u']
#
# 	def __call__(self, module, input):
# 		self.compute_weight(module)
#
#
# def spectral_norm(module, name='weight', dim=0):
# 	SpectralNorm.apply(module, name)
# 	return module