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
        self.gpu = hyperparameters['gpu']
        cudnn.benchmark = True

        # init G
        if hyperparameters["model"] == 'DeepLab':
            self.model = Res_Deeplab(num_classes=hyperparameters["num_classes"])


        self.model.train()
        self.model.cuda(self.gpu)


        # init D
        self.model_D = FCDiscriminator(num_classes=hyperparameters['num_classes'])
        self.model_D.train()
        self.model_D.cuda(self.gpu)

        # Setup the optimizers
        self.lr_g = hyperparameters['lr_g']
        self.lr_d = hyperparameters['lr_d']

        momentum = hyperparameters['momentum']
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        weight_decay = hyperparameters['weight_decay']

        self.optimizer_G = optim.SGD(self.model.parameters(),
                                     lr=self.lr_, momentum=momentum, weight_decay=weight_decay)
        self.optimizer_D = optim.Adam(self.model_D.parameters(), lr=self.lr_d, betas=(beta1, beta2))

        # dynamic adjust lr setting
        self.decay_power = hyperparameters['decay_power']

        # for log or for save model
        self.loss_d_value = 0
        self.loss_source_value = 0
        self.loss_target_value = 0
        self.num_steps = 0

        # for discriminator
        self.adv_loss_opt = hyperparameters['dis']['adv_loss_opt']
        self.source_label, self.target_label = 0, 1
        self.source_image = None
        self.target_image = None
        self.inter_mini = nn.Upsample(size=(self.input_size_target[1], self.input_size_target[0]), align_corners=False,
                                    mode='bilinear')

    def forward(self, images):
        self.eval()
        images = Variable(images).cuda(self.gpu)
        predict_seg = self.model(images)
        self.train()
        return predict_seg

    def gen_source_update(self, images, labels):
        """
                Input source domain image and compute segmentation loss.

                :param images:
                :param labels:
                :return:
                """
        # Disable D backpropgation, we only train G
        for param in self.model_D.parameters():
            param.requires_grad = False

        # load data
        images = Variable(images).cuda(self.gpu)
        labels = Variable(labels).cuda(self.gpu)

        # get predict output
        pred_source_real = self.model(images)

        # resize to source size
        interp = nn.Upsample(size=(self.input_size[1], self.input_size[0]), align_corners=False, mode='bilinear')
        pred_source_real = interp(pred_source_real)

        # in source domain compute segmentation loss
        seg_loss = self._compute_seg_loss(pred_source_real, labels)
        # proper normalization
        seg_loss /= self.iter_size
        seg_loss.backward()

        # save image for discriminator use
        self.source_image = pred_source_real

        # record log
        self.loss_source_value += seg_loss.data.cpu().numpy()

    def gen_target_update(self, images):
        """
                 Input target domain image and compute adversarial loss.

                :param images:
                :return:
                """
        # Disable D backpropgation, we only train G
        for param in self.model_D.parameters():
            param.requires_grad = False

        # load data
        images = Variable(images).cuda(self.gpu)

        # get predict output
        pred_target_fake = self.model(images)

        # resize to target size
        interp_target = nn.Upsample(size=(self.input_size_target[1], self.input_size_target[0]), align_corners=False,
                                    mode='bilinear')
        pred_target_fake = interp_target(pred_target_fake)

        # d_out_fake = model_D1(F.softmax(pred_target_fake), inter_mini(F.softmax(pred_target_fake)))
        d_out_fake = self.model_D1(F.softmax(pred_target_fake))
        # compute loss function
        adv_loss = self._compute_adv_loss_fake(d_out_fake, loss_opt=self.adv_loss_opt)
        # proper normalization
        adv_loss /= self.iter_size
        adv_loss.backward()

        # save image for discriminator use
        self.target_image = pred_target_fake

        # record log
        self.loss_target_value += adv_loss.data.cpu().numpy()

    def dis_update(self):
        """
                use [gen_source_update / gen_target_update]'s image to discriminator,
                so you  don' t need to give any parameter
                """
        # Enable D backpropgation, train source G and train D
        for param in self.model_D.parameters():
            param.requires_grad = True
        # we don't train target's G weight, we only train source'G
        self.target_image = self.target_image.detach()
        # compute loss function
        loss_real = self._compute_adv_loss_real(self.source_image, self.adv_loss_opt)
        loss_fake = self._compute_adv_loss_fake(self.target_image, self.adv_loss_opt)
        # to image
        dis_loss = (loss_real + loss_fake) / self.adv_loss_opt
        dis_loss.backward()

        # record log
        self.loss_d_value += dis_loss.data.cpu().numpy()

    def show_each_loss(self):
        print("iter = {0:8d}/{1:8d}, loss_G_source_1 = {2:.3f} loss_G_adv1 = {3:.3f} loss_D1 = {4:.3f}".format(
            self.i_iter, self.num_steps, self.loss_source_value, float(self.loss_target_value), float(self.loss_d_value)))

    def _compute_adv_loss_real(self, d_out_real, loss_opt="bce", label=0):
        """
                compute adversarial loss function, can choose loss opt
                :param d_out_fake:
                :param loss_opt:  [wgan-gp / hinge / bce]
                :param label:
                :return:
                """

        if loss_opt == 'wgan-gp':
            d_loss_real = - d_out_real.mean()
        elif loss_opt == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        elif loss_opt == 'bce':
            bce_loss = torch.nn.BCEWithLogitsLoss()
            d_loss_real = bce_loss(d_out_real,
                                   Variable(torch.FloatTensor(d_out_real.data.size()).fill_(source_label)).cuda(
                                       self.gpu))
        return d_loss_real

    def _compute_adv_loss_fake(self, d_out_fake, loss_opt="bce", label=1):
        """
                compute adversarial loss function, can choose loss opt
                :param d_out_fake:
                :param loss_opt:  [wgan-gp / hinge / bce]
                :param label:
                :return:
                """

        if loss_opt == 'wgan-gp':
            d_loss_fake = - d_out_fake.mean()
        elif loss_opt == 'hinge':
            d_loss_fake = - d_out_fake.mean()
        elif loss_opt == 'bce':
            bce_loss = torch.nn.BCEWithLogitsLoss()
            d_loss_fake = bce_loss(d_out_fake,
                                   Variable(torch.FloatTensor(d_out_fake.data.size()).fill_(label)).cuda(
                                       self.gpu))
        return d_loss_fake

    def _compute_seg_loss(self, pred, label):
        """
                This function returns cross entropy loss for semantic segmentation
                """
        # out shape batch_size x channels x h x w -> batch_size x channels x h x w
        # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
        label = Variable(label.long()).cuda(self.gpu)
        criterion = CrossEntropy2d().cuda(self.gpu)

        return criterion(pred, label)

    def _lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** power)

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

    def update_learning_rate(self, num_steps):
        if self.optimizer_D is not None:
            self.optimizer_D.step()
            self.optimizer_D.zero_grad()
            self._adjust_learning_rate_D(self.optimizer_D, num_steps)

        if self.optimizer_G is not None:
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()
            self._adjust_learning_rate_G(self.optimizer_G, num_steps)

        self.num_steps = num_steps
    def snapshot_image_source_domain(self, images, labels):
        """
                check model training status,
                will output image to config["image_save_dir"]
                :param images:
                :param labels:
                :return:
                """

        # save label
        label_name = os.path.join("data", "GTA5", "labels", names[0])
        save_name = 'check_output/Image_source_domain_seg/%s_label.png' % (i_iter)
        shutil.copyfile(label_name, save_name)

        target_name = os.path.join("data", "Cityscapes", "data", "leftImg8bit", "train", target_name[0])
        save_name = 'check_output/Image_target_domain_seg/%s_label.png' % (i_iter)
        shutil.copyfile(target_name, save_name)

        # save output image
        output_to_image(pred_source_real).save('check_output/Image_source_domain_seg/%s.png' % (i_iter))
        output_to_image(pred_target_fake).save('check_output/Image_target_domain_seg/%s.png' % (i_iter))

    def save_model(self, snapshot_save_dir="./model_save"):
        """
                save model to .pth file
                will output model to config["snapshot_save_dir"]
                """
        print('taking pth in shapshot dir ...')
        torch.save(self.model.state_dict(), osp.join(snapshot_save_dir, 'GTA5_' + str(self.i_iter) + '.pth'))
        torch.save(self.model_D1.state_dict(), osp.join(snapshot_save_dir, 'GTA5_' + str(self.i_iter) + '_D1.pth'))

    def restore(self, model_name=None, num_classes=19, restore_from=None):
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