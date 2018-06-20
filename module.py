import pickle
import sys
import os
import os.path as osp
import random
import shutil
import pdb

import argparse
import scipy.misc
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
from torch.optim import lr_scheduler

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# from model.deeplab_multi import Res_Deeplab
from model.deeplab_single import Res_Deeplab
from model.discriminator import FCDiscriminator
from model.xiao_discriminator import XiaoDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
class AdaptSeg_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(AdaptSeg_Trainer, self).__init__()
        # input size setting
        self.input_size = (hyperparameters["input_size_h"], hyperparameters["input_size_w"])
        self.input_size_target = (hyperparameters["input_target_size_h"], hyperparameters["input_target_size_w"])

        # training setting
        self.num_steps = hyperparameters["num_steps"]

        # log setting
        configure(hyperparameters['config_path'])

        # cuda setting
        gpu = hyperparameters['gpu']
        cudnn.benchmark = True

        # init G
        if hyperparameters["model"] == 'DeepLab':
            self.model = Res_Deeplab(num_classes=num_classes)

        if hyperparameters["restore"]:
            self.restore_opt(hyperparameters["model"], hyperparameters["num_classes"], hyperparameters["restore_from"])

        self.model.train()
        self.model.cuda(gpu)


        # init D
        self.model_D = FCDiscriminator(num_classes=hyperparameters['num_classes'])
        self.model_D.train()
        self.model_D.cuda(gpu)

        # Setup the optimizers
        self.lr_g = hyperparameters['lr_g']
        self.lr_d = hyperparameters['lr_d']
        self.power = hyperparameters['lr_d']

        momentum = hyperparameters['momentum']
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        weight_decay = hyperparameters['weight_decay']

        self.optimizer_G = optim.SGD(self.model.parameters(),
                              lr=lr_g, momentum=momentum, weight_decay=weight_decay)
        self.optimizer_D = optim.Adam(self.model_D.parameters(), lr=lr_d, betas=(beta1, beta2))

        # dynamic adjust lr setting
        self.decay_power = hyperparameters['decay_power']

    def _lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def _adjust_learning_rate_D(self, optimizer, i_iter):
        # TODO : check optimizer param_groups
        lr = self._lr_poly(self.lr_d, i_iter, self.num_steps, self.decay_power)
        print("_adjust_learning_rate_D param_groups =", len(optimizer.param_groups))
        for i, group in enumerate(optimizer.param_groups):
            optimizer.param_groups[i]['lr'] = lr

    def _adjust_learning_rate_G(self, optimizer, i_iter):
        # TODO : check optimizer param_groups
        lr = self._lr_poly(self.lr_g, i_iter, self.num_steps, self.decay_power)
        print("_adjust_learning_rate_G param_groups =", len(optimizer.param_groups))
        for i, group in enumerate(optimizer.param_groups):
            optimizer.param_groups[i]['lr'] = lr


    def update_learning_rate(self, i_iter):
        if self.optimizer_D is not None:
            self.dis_scheduler.step()
            self._adjust_learning_rate_D(self.optimizer_D, i_iter)
        if self.self.optimizer_G is not None:
            self.gen_scheduler.step()
            self._adjust_learning_rate_G(self.optimizer_G, i_iter)


    def restore_opt(self, model_name, num_classes, restore_from):
        if model_name == 'DeepLab':
            self.model = Res_Deeplab(num_classes=num_classes)
            print("check restore from", restore_from)
            if restore_from[:4] == 'http':
                saved_state_dict = model_zoo.load_url(restore_from)
            else:
                saved_state_dict = torch.load(restore_from)
            new_params = self.model.state_dict().copy()
            for i in saved_state_dict:
                # print(i)
                # Scale.layer5.conv2d_list.3.weight
                i_parts = i.split('.')
                # print i_parts
                if not num_classes == 19 or not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                    # print(i_parts)
            # new_params = saved_state_dict
            self.model.load_state_dict(new_params)