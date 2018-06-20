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
snapshot_save_iter = config["snapshot_save_iter"]
snapshot_save_dir = config["snapshot_save_dir"]
image_save_iter = config["image_save_iter"]
image_save_dir = config["image_save_dir"]

# train_writer = tensorboardX.SummaryWriter(os.path.join(log_path, "logs", test_summary))

# data loader
train_loader, target_loader = get_all_data_loaders(config)

# model init
trainer = AdaptSeg_Trainer(config)

if config["restore"]:
    trainer.restore(model_name=config["model"], num_classes=config["num_classes"], restore_from=config["restore_from"])

# Start training
while True:
    for i_iter, (train_batch, target_batch) in enumerate(zip(train_loader, target_loader)):
        # ====================== #
        #   Main training code   #
        # ====================== #

        # train G use source image
        images, labels, _, names = train_batch
        trainer.gen_source_update(images, labels)

        # train G use target image
        images, _, _, target_name = target_batch
        trainer.gen_target_update(images)

        # train discriminator use prior generator image
        trainer.dis_update()

        # update loss function
        trainer.update_learning_rate(i_iter)

        # show log
        trainer.show_each_loss()

        # save image to check
        if i_iter % image_save_iter == 0:
            trainer.(snapshot_save_dir=image_save_dir)


        # save checkpoint .pth
        if i_iter % snapshot_save_iter == 0:
            trainer.save_model(snapshot_save_dir=snapshot_save_dir)

        # save final model .pth
        if i_iter == num_steps - 1:
            trainer.save_model(snapshot_save_dir=snapshot_save_dir)



