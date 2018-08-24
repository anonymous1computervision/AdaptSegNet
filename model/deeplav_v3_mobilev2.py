import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint_sequential

import layers


# create model
class MobileNetv2_DeepLabv3(nn.Module):
    """
    A Convolutional Neural Network with MobileNet v2 backbone and DeepLab v3 head
        used for Semantic Segmentation on Cityscapes dataset
    """

    """######################"""
    """# Model Construction #"""
    """######################"""

    def __init__(self, pre_trained_from=None, resume_from=None):
        super(MobileNetv2_DeepLabv3, self).__init__()
        self.resume_from = resume_from
        self.pre_trained_from = pre_trained_from
        self.epoch = 0
        self.init_epoch = 0
        self.ckpt_flag = False
        self.train_loss = []
        self.val_loss = []
        s = [2, 1, 2, 2, 2, 1, 1]  # stride of each conv stage
        t = [1, 1, 6, 6, 6, 6, 6]  # expansion factor t
        n = [1, 1, 2, 3, 4, 3, 3]  # number of repeat time
        c = [32, 16, 24, 32, 64, 96, 160]  # output channel of each conv stage
        down_sample_rate = 32  # classic down sample rate
        aspp = (6, 12, 18)
        multi_grid = (1, 2, 4)
        output_stride = 16
        num_class = 19

        # build network
        block = []

        # conv layer 1
        block.append(nn.Sequential(nn.Conv2d(3, c[0], 3, stride=s[0], padding=1, bias=False),
                                   nn.BatchNorm2d(c[0]),
                                   # nn.Dropout2d(dropout_prob, inplace=True),
                                   nn.ReLU6()))

        # conv layer 2-7
        for i in range(6):
            block.extend(layers.get_inverted_residual_block_arr(c[i], c[i+1],
                                                                t=t[i+1], s=s[i+1],
                                                                n=n[i+1]))

        # dilated conv layer 1-4
        # first dilation=rate, follows dilation=multi_grid*rate
        rate = down_sample_rate // output_stride
        block.append(layers.InvertedResidual(c[6], c[6],
                                             t=t[6], s=1, dilation=rate))
        for i in range(3):
            block.append(layers.InvertedResidual(c[6], c[6],
                                                 t=t[6], s=1, dilation=rate*multi_grid[i]))

        # ASPP layer

        block.append(layers.ASPP_plus(c=c, aspp=aspp))

        # final conv layer
        block.append(nn.Conv2d(256, num_class, 1))

        # bilinear upsample
        block.append(nn.Upsample(scale_factor=output_stride, mode='bilinear', align_corners=False))

        self.network = nn.Sequential(*block).cuda()

        # initialize
        self.initialize()

        # load data
        self.load_checkpoint()
        self.load_model()

    def load_checkpoint(self):
        """
        Load checkpoint from given path
        """
        if self.resume_from is not None and os.path.exists(self.resume_from):
            try:
                print('Loading Checkpoint at %s' % self.resume_from)
                ckpt = torch.load(self.resume_from)
                self.epoch = ckpt['epoch']
                try:
                    self.train_loss = ckpt['train_loss']
                    self.val_loss = ckpt['val_loss']
                except:
                    self.train_loss = []
                    self.val_loss = []
                self.network.load_state_dict(ckpt['state_dict'])
                print('network.load_state_dict Loaded!')

                self.opt.load_state_dict(ckpt['optimizer'])
                print('Checkpoint Loaded!')
                print('Current Epoch: %d' % self.epoch)
                self.ckpt_flag = True
            except:
                print('Cannot load checkpoint from %s. Start loading pre-trained model......' % self.resume_from)
        else:
            print('Checkpoint do not exists. Start loading pre-trained model......')

    def load_model(self):
        """
        Load ImageNet pre-trained model into MobileNetv2 backbone, only happen when
            no checkpoint is loaded
        """
        if self.ckpt_flag:
            print('Skip Loading Pre-trained Model......')
        else:
            if self.pre_trained_from is not None and os.path.exists(self.pre_trained_from):
                try:
                    print('Loading Pre-trained Model at %s' % self.pre_trained_from)
                    pretrain = torch.load(self.pre_trained_from)
                    self.network.load_state_dict(pretrain)
                    print('Pre-trained Model Loaded!')
                except:
                    print('Cannot load pre-trained model. Start training......')
            else:
                print('Pre-trained model do not exits. Start training......')

    """#############"""
    """# Utilities #"""
    """#############"""

    def initialize(self):
        """
        Initializes the model parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# """ TEST """
# if __name__ == '__main__':
#     params = CIFAR100_params()
#     params.dataset_root = '/home/ubuntu/cifar100'
#     net = MobileNetv2(params)
#     net.save_checkpoint()