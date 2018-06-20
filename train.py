"""
Code concept modify from AdaptSegNet
    - Learning to Adapt Structured Output Space for Semantic Segmentation
    - https://arxiv.org/abs/1802.10349
    - https://github.com/wasidennis/AdaptSegNet

Code style modify Modify from MUNIT
    - https://github.com/NVlabs/MUNIT

Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
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
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorboardX

# from model.deeplab_multi import Res_Deeplab
from model.deeplab_single import Res_Deeplab
from model.discriminator import FCDiscriminator
from model.xiao_discriminator import XiaoDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from module import AdaptSeg_Trainer

from util import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images

# cuda setting
cudnn.benchmark = True

# config setting
CONFIG_PATH = "./config/default.yaml"
config = get_config(CONFIG_PATH)

# training setting
num_steps = config['num_steps']

# log setting
log_path = config["log_path"]
test_summary = config["test_summary"]
# train_writer = tensorboardX.SummaryWriter(os.path.join(log_path, "logs", test_summary))

# data loader
train_loader, target_loader = get_all_data_loaders(config)

# model init
trainer = AdaptSeg_Trainer(config)

# Start training
while True:
    for it, (train_batch, target_batch) in enumerate(zip(train_loader, target_loader)):
        images, labels, _, names = train_batch
        images, _, _, target_name = target_batch

        trainer.update_learning_rate()

        # Main training code
        trainer.gen_update(images_a, images_b, config)
        trainer.dis_update(images_a, images_b, config)

