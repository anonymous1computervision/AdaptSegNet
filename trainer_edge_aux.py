import os
import shutil

# from tensorboard_logger import configure, log_value
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

# from model.deeplab_multi import Res_Deeplab
from model.deeplab_single import Res_Deeplab
from model.deeplab_single_add_edge import Res_Deeplab as Res_Deeplab_Edge
# from model.deeplab_single_add_edge_fuse import Res_Deeplab as Res_Deeplab_Edge_Fuse

from model.deeplab_single_add_edge_v2 import Res_Deeplab as Res_Deeplab_Edge_v2

from model.deeplav_v3_xception import DeepLabv3_plus
import model.fc_densenet as fc_densenet
from model.sp_discriminator import SP_FCDiscriminator
from model.sp_attn_discriminator import SP_ATTN_FCDiscriminator
from model.sp_one_attn_discriminator import SP_One_ATTN_FCDiscriminator
from model.sp_cgan_discriminator import SP_CGAN_FCDiscriminator
from model.sp_coord_discriminator import SP_Coord_FCDiscriminator
from model.sp_feature_discriminator import SP_Feature_FCDiscriminator

from model.gated_discriminator import Gated_Discriminator
# from model.gated_sa_discriminator import Gated_Self_Attn_Discriminator

from model.gated_coord_discriminator import Gated_Coord_Discriminator
from model.gated_coord_hinge_discriminator import Gated_Coord_Hinge_Discriminator

# from model.gated_first2layer_discriminator import Gated_First2Layer_Discriminator
# from model.gated_first2layer_hinge_discriminator import Gated_First2Layer_Hinge_Discriminator

from model.sp_aspp_discriminator import SP_ASPP_FCDiscriminator

from util.loss import CrossEntropy2d

class AdaptSeg_Edge_Aux_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(AdaptSeg_Edge_Aux_Trainer, self).__init__()
        self.hyperparameters = hyperparameters
        # input size setting
        self.input_size = (hyperparameters["input_size_h"], hyperparameters["input_size_w"])
        # self.input_size = (760, 760)

        self.input_size_target = (hyperparameters["input_target_size_h"], hyperparameters["input_target_size_w"])

        # training setting
        self.num_steps = hyperparameters["num_steps"]



        # log setting
        # configure(hyperparameters['config_path'])

        # cuda setting
        self.gpu = hyperparameters['gpu']
        cudnn.benchmark = True

        # todo: remove this line without dev version
        assert hyperparameters["model"] == 'DeepLabEdge', True

        # init G
        # if hyperparameters["model"] == 'DeepLab':
        #     self.model = Res_Deeplab(num_classes=hyperparameters["num_classes"])
        # elif hyperparameters["model"] == 'FC-DenseNet':
        #     self.model = fc_densenet.FCDenseNet57(hyperparameters["num_classes"])
        #     print("use fc densenet model")
        # elif hyperparameters["model"] == 'DeepLabEdge':
        self.model = Res_Deeplab_Edge(num_classes=hyperparameters["num_classes"])
        # self.model = Res_Deeplab_Edge_Fuse(num_classes=hyperparameters["num_classes"])

        # self.model = Res_Deeplab_Edge_v2(num_classes=hyperparameters["num_classes"])
        # print("use DeepLabEdge v2 model")
        # print("use DeepLabEdge v2 model")
        # print("use DeepLabEdge v2 model")

        # print("use DeepLabEdge model")
        # elif hyperparameters["model"] == 'DeepLab_v3_plus':
        #     self.model = DeepLabv3_plus(nInputChannels=3,
        #                                 n_classes=hyperparameters['num_classes'],
        #                                 pretrained=True,
        #                                 _print=True)
        #     print("use DeepLab_v3_plus model")
        # init D
        # self.model_D = FCDiscriminator(num_classes=hyperparameters['num_classes'])
        # self.model_D = SP_FCDiscriminator(num_classes=hyperparameters['num_classes'])
        # self.model_D = SP_CGAN_FCDiscriminator(num_classes=hyperparameters['num_classes'])
        # self.model_D = SP_Coord_FCDiscriminator(num_classes=hyperparameters['num_classes'])

        self.model_D = SP_ATTN_FCDiscriminator(num_classes=hyperparameters['num_classes'])
        # self.model_D = SP_One_ATTN_FCDiscriminator(num_classes=hyperparameters['num_classes'])

        # self.model_D = SP_ASPP_FCDiscriminator(num_classes=hyperparameters['num_classes'])

        # self.model_D = SP_ATTN_FCDiscriminator(num_classes=hyperparameters['num_classes'])

        # self.model_D = SP_FCDiscriminator(num_classes=hyperparameters['num_classes'])
        # self.model_D = SP_Feature_FCDiscriminator(num_classes=hyperparameters['num_classes'])
        # self.model_D = SP_Feature_FCDiscriminator(num_classes=hyperparameters['num_classes']+1)
        # self.model_D = SP_Feature_FCDiscriminator(num_classes=hyperparameters['num_classes'])
        self.model_D_foreground = Gated_Discriminator(num_classes=hyperparameters['num_classes'] + 1)
        # self.model_D_foreground = Gated_Self_Attn_Discriminator(num_classes=hyperparameters['num_classes'] + 1)

        # self.model_D_foreground = Gated_Discriminator(num_classes=hyperparameters['num_classes'] + 1)
        # self.model_D_foreground = Gated_Coord_Discriminator(num_classes=hyperparameters['num_classes'] + 1)
        # self.model_D_foreground = Gated_Coord_Hinge_Discriminator(num_classes=hyperparameters['num_classes'] + 1)


        # self.model_D_foreground = Gated_First2Layer_Hinge_Discriminator(num_classes=hyperparameters['num_classes'] + 1)

        # self.model_D_foreground = Gated_Hinge_Discriminator(num_classes=hyperparameters['num_classes'] + 1)

        # self.model_D_foreground = Partial_Discriminator(num_classes=hyperparameters['num_classes'])

        # self.model_D_ = SP_FCDiscriminator(num_classes=3+1)

        # self.model_D_cgan = XiaoCganDiscriminator(num_classes=hyperparameters['num_classes'])
        # self.model_D = XiaoCganResAttnDiscriminator(num_classes=hyperparameters['num_classes'])

        # +1 means add edge info
        # self.model_D = SP_FCDiscriminator(num_classes=hyperparameters['num_classes']+1)
        # self.model_D = SP_ATTN_FCDiscriminator(num_classes=hyperparameters['num_classes']+1)

        # =====================
        # D focus in foreground
        # =====================
        # 11 -> person: 47.65
        # 12 - > rider: 32.78
        # 13 - > car: 78.97
        # 14 - > truck: 37.58
        # 15 - > bus: 32.1
        # 16 - > train: 2.48
        # 17 - > motocycle: 0.0
        # 18 - > bicycle: 4.52
        # self.model_D_foreground = SP_FCDiscriminator(num_classes=8)

        # self.model_D = XiaoAttentionDiscriminator(num_classes=hyperparameters['num_classes'])
        # self.model_D = XiaoPretrainAttentionDiscriminator(num_classes=hyperparameters['num_classes'])
        # self.model_D = SP_ATTN_FCDiscriminator(num_classes=hyperparameters['num_classes'])
        # self.model_D = SP_ASPP_FCDiscriminator(num_classes=hyperparameters['num_classes'])

        self.model.train()
        self.model.cuda(self.gpu)
        self.model_D.train()
        self.model_D.cuda(self.gpu)
        self.model_D_foreground.train()
        self.model_D_foreground.cuda(self.gpu)

        # for dynamic adjust lr setting
        self.decay_power = hyperparameters['decay_power']

        # init optimizer
        self.lr_g = hyperparameters['lr_g']
        self.lr_d = hyperparameters['lr_d']

        self.momentum = hyperparameters['momentum']
        self.weight_decay = hyperparameters['weight_decay']
        self.beta1 = hyperparameters['beta1']
        self.beta2 = hyperparameters['beta2']
        self.D_beta1 = hyperparameters['D_beta1']
        self.D_beta2 = hyperparameters['D_beta2']
        self.init_opt()

        # for [log / check output]
        # self.loss_d_value = 0
        # self.loss_d_foreground_value = 0
        self.loss_source_value = 0
        self.loss_target_value = 0
        self.loss_edge_value = 0

        self.loss_target_foreground_value = 0
        # self.loss_target_weakly_seg_value = 0
        # self.loss_d_attn_value = 0

        self.i_iter = 0
        self.source_label_path = None
        self.target_image_path = None

        # for generator
        self.lambda_seg = hyperparameters['gen']['lambda_seg']
        self.lambda_adv_target = hyperparameters['gen']['lambda_adv_target']
        self.lambda_adv_foreground = hyperparameters['gen']['lambda_adv_foreground']
        self.lambda_adv_edge = hyperparameters['gen']['lambda_adv_edge']

        self.decay_power = hyperparameters['decay_power']

        # for discriminator
        self.adv_loss_opt = hyperparameters['dis']['adv_loss_opt']
        # self.lambda_attn = hyperparameters['dis']['lambda_attn']

        self.mini_factor_size = 8
        self.source_image = None
        self.target_image = None
        self.saliency_mask = None
        self.pred_real_edge = None
        self.pred_fake_edge = None
        # self.inter_mini = nn.Upsample(size=self.input_size_target, align_corners=False,
        #                               mode='bilinear')

        self.loss_names = ['Seg', 'Edge_Seg', 'Global_GAN_dis', 'Global_GAN_adv', 'Foreground_GAN_dis', 'Foreground_GAN_adv']
        self.loss_dict = {k: 0 for k in self.loss_names}

    def init_opt(self):

        # todo: change to origin
        # self.optimizer_G = optim.SGD(self.model.optim_parameters_lr(self.lr_g),
        #                       lr=self.lr_g, momentum=self.momentum, weight_decay=self.weight_decay)
        # self.optimizer_G.zero_grad()
        self.optimizer_G = optim.SGD([p for p in self.model.parameters() if p.requires_grad],
                                     lr=self.lr_g, momentum=self.momentum, weight_decay=self.weight_decay)
        # self.optimizer_G = optim.SGD([p for p in self.model.parameters() if p.requires_grad],
        #                              lr=self.lr_g, momentum=momentum, weight_decay=weight_decay)
        self.optimizer_G.zero_grad()
        self._adjust_learning_rate_G(self.optimizer_G, 0)

        self.optimizer_D = optim.Adam([p for p in self.model_D.parameters() if p.requires_grad],
                                      lr=self.lr_d, betas=(self.beta1, self.beta2))
        self.optimizer_D.zero_grad()
        self._adjust_learning_rate_D(self.optimizer_D, 0)

        self.optimizer_D_foreground = optim.Adam([p for p in self.model_D_foreground.parameters() if p.requires_grad],
                                                 lr=self.lr_d, betas=(self.D_beta1, self.D_beta2))
        self.optimizer_D_foreground.zero_grad()
        self._adjust_learning_rate_D(self.optimizer_D_foreground, 0)

    def forward(self, images):
        self.eval()
        # pdb.set_trace()
        predict_seg, _ = self.model(images)
        self.train()
        return predict_seg

    # def gen_source_update(self, images, labels, label_path=None):
    #     """
    #             Input source domain image and compute segmentation loss.
    #
    #             :param images:
    #             :param labels:
    #             :param label_path: just for save path to record model predict, use in  snapshot_image_save function
    #
    #             :return:
    #             """
    #     self.optimizer_G.zero_grad()
    #
    #     # Disable D backpropgation, we only train G
    #     for param in self.model_D.parameters():
    #         param.requires_grad = False
    #
    #     for param in self.model_D_foreground.parameters():
    #         param.requires_grad = False
    #
    #     self.source_label_path = label_path
    #     # get predict output
    #     pred_source_real, _ = self.model(images)
    #
    #     # resize to source size
    #     interp = nn.Upsample(size=self.input_size, align_corners=False, mode='bilinear')
    #
    #     pred_source_real = interp(pred_source_real)
    #
    #     # in source domain compute segmentation loss
    #     seg_loss = self._compute_seg_loss(pred_source_real, labels)
    #
    #     seg_loss = self.lambda_seg*seg_loss
    #     seg_loss.backward()
    #
    #     # update loss
    #     self.optimizer_G.step()
    #
    #     self.source_image = pred_source_real.detach()
    #
    #     self.source_input_image = images.detach()
    #
    #     # record log
    #     self.loss_dict['Seg'] += seg_loss.data.cpu().numpy()
    #     del seg_loss
    #     self._gen_source_attn(images, labels)
    #
    # def _gen_source_attn(self, images, labels):
    #     """
    #             Input source domain image and compute segmentation loss.
    #
    #             :param images:
    #             :param labels:
    #             :param label_path: just for save path to record model predict, use in  snapshot_image_save function
    #
    #             :return:
    #             """
    #     self.optimizer_G.zero_grad()
    #
    #     _, pred_source_edge = self.model(images)
    #
    #     # resize to source size
    #     interp = nn.Upsample(size=self.input_size, align_corners=False, mode='bilinear')
    #     pred_source_edge = interp(pred_source_edge)
    #
    #     bce_loss = nn.BCEWithLogitsLoss()
    #
    #     attn_loss = bce_loss(pred_source_edge,
    #                          self.get_foreground_attention(
    #                              labels.view(labels.shape[0], -1, labels.shape[1], labels.shape[2]).cuda()))
    #     attn_loss = self.lambda_adv_edge*attn_loss
    #     attn_loss.backward()
    #
    #     # update loss
    #     self.optimizer_G.step()
    #     self.pred_real_edge = nn.Sigmoid()(pred_source_edge).detach()
    #
    #     self.loss_dict['Edge_Seg'] += self.lambda_adv_edge * attn_loss.data.cpu().numpy()
    #     del attn_loss

    def gen_source_update(self, images, labels, label_path=None):
        """
                Input source domain image and compute segmentation loss.

                :param images:
                :param labels:
                :param label_path: just for save path to record model predict, use in  snapshot_image_save function

                :return:
                """
        self.optimizer_G.zero_grad()

        # Disable D backpropgation, we only train G
        for param in self.model_D.parameters():
            param.requires_grad = False

        for param in self.model_D_foreground.parameters():
            param.requires_grad = False

        self.source_label_path = label_path

        # print("images shape =", images.shape)
        # get predict output
        pred_source_real, pred_source_edge = self.model(images)

        # resize to source size
        interp = nn.Upsample(size=self.input_size, align_corners=False, mode='bilinear')


        # old version
        # interp = nn.Upsample(size=self.input_size, align_corners=True, mode='bilinear')
        pred_source_real = interp(pred_source_real)
        # interp_source_mini = nn.Upsample(size=(int(self.input_size[0]/8), int(self.input_size[1]/8)), align_corners=False,
        #                             mode='bilinear')
        # interp_source_mini = nn.Upsample(
        #     size=(int(self.input_size[0] / self.mini_factor_size), int(self.input_size[1] / self.mini_factor_size)),
        #     align_corners=False,
        #     mode='bilinear')
        # self.pred_source_edge_mini = interp_source_mini(pred_source_edge).detach()
        pred_source_edge = interp(pred_source_edge)

        # in source domain compute segmentation loss
        seg_loss = self._compute_seg_loss(pred_source_real, labels)

        # in source domain compute edge loss
        # edge_loss = self._compute_edge_loss(pred_source_edge, labels)
        # seg_loss = self.lambda_seg * seg_loss
        # seg_loss.backward()

        bce_loss = nn.BCEWithLogitsLoss()
        # # Todo : here use softmax maybe wrong
        # edge_loss = bce_loss(pred_source_edge,
        #                       self.label_get_edges(labels.view(labels.shape[0], 1, labels.shape[1], labels.shape[2]).cuda()))
        # attn_loss = bce_loss(pred_source_edge,
        #                      self.get_foreground_attention(
        #                          labels.view(labels.shape[0], -1, labels.shape[1], labels.shape[2]).cuda()))
        attn_loss = bce_loss(pred_source_edge,
                             self.get_foreground_attention(
                                 labels.view(labels.shape[0], -1, labels.shape[1], labels.shape[2]).cuda()))
        # attn_loss = self.lambda_adv_edge * attn_loss
        # attn_loss.backward()
        # proper normalization
        # seg_loss = self.lambda_seg * seg_loss
        seg_loss = self.lambda_seg*seg_loss + self.lambda_adv_edge*attn_loss
        seg_loss.backward()
        # seg_loss = self.lambda_seg*seg_loss + self.lambda_adv_edge*edge_loss
        # seg_loss.backward()

        # update loss
        self.optimizer_G.step()

        # save image for discriminator use
        self.source_image = pred_source_real.detach()
        self.pred_real_edge = nn.Sigmoid()(pred_source_edge).detach()
        # self.pred_real_edge = pred_source_edge.detach()

        self.source_input_image = images.detach()

        # record log
        self.loss_dict['Seg'] += seg_loss.data.cpu().numpy()
        self.loss_dict['Edge_Seg'] += self.lambda_adv_edge * attn_loss.data.cpu().numpy()
        del attn_loss, seg_loss
        # self.loss_source_value += seg_loss.data.cpu().numpy()
        # self.loss_edge_value += self.lambda_adv_edge*edge_loss.data.cpu().numpy()
        # self.loss_edge_value += self.lambda_adv_edge * attn_loss.data.cpu().numpy()



    def gen_target_update(self, images, image_path):
        """
                 Input target domain image and compute adversarial loss.

                :param images:
                :param image_path: just for save path to record model predict, use in  snapshot_image_save function
                :return:
                """

        self.optimizer_G.zero_grad()

        # Disable D backpropgation, we only train G
        for param in self.model_D.parameters():
            param.requires_grad = False

        for param in self.model_D_foreground.parameters():
            param.requires_grad = False

        self.target_image_path = image_path

        # get predict output
        pred_target_fake, pred_target_edge = self.model(images)

        # resize to target size
        interp_target = nn.Upsample(size=self.input_size_target, align_corners=False,
                                    mode='bilinear')
        pred_target_fake = interp_target(pred_target_fake)

        pred_target_edge = interp_target(pred_target_edge)

        # cobime predict and use predict output get edge
        # net_input = torch.cat((F.softmax(pred_target_fake), pred_target_edge), dim=1)
        net_input = F.softmax(pred_target_fake, dim=1)
        # print("net input shape =", net_input.shape)
        # d_out_fake, _ = self.model_D(F.softmax(pred_target_fake), label=images)
        # d_out_fake, _ = self.model_D(F.softmax(net_input), label=images)
        self.pred_fake_edge = nn.Sigmoid()(pred_target_edge).detach()

        # d_out_fake, _ = self.model_D(net_input, label=self.pred_fake_edge)
        d_out_fake, _ = self.model_D(net_input, label=None)
        d_out_foreground_fake, _ = self.model_D_foreground(net_input, label=self.pred_fake_edge)

        #
        # net_input = torch.cat((net_input, pred_target_edge), dim=1)
        # d_out_fake, _ = self.model_D_cgan(net_input, label=images)

        # d_out_fake, _ = self.model_D(net_input, label=self.pred_target_edge_mini)
        # d_out_fake, _ = self.model_D(net_input, label=self.pred_target_edge_mini-0.5)
        # compute loss function
        # wants to fool discriminator
        # adv_loss = self.loss_hinge_gen(d_out_fake)

        # loss_adv_foreground = self.loss_hinge_gen(d_out_foreground_fake)

        if self.adv_loss_opt == "hinge":
            loss_adv = self.loss_hinge_gen(d_out_fake)
            loss_adv_foreground = self.loss_hinge_gen(d_out_foreground_fake)
        else:
            loss_adv = self._compute_adv_loss_real(d_out_fake)
            loss_adv_foreground = self._compute_adv_loss_real(d_out_foreground_fake)


        # adv_loss = self.loss_hinge_gen(d_out_fake) + 0.5 * loss_adv_foreground
        # adv_loss = self.loss_hinge_gen(d_out_fake) + loss_adv_foreground
        # adv_loss = self.loss_hinge_gen(d_out_fake) + 2*loss_adv_foreground

        loss = loss_adv + self.lambda_adv_foreground * loss_adv_foreground

        # adv_loss = self.loss_hinge_gen(d_out_fake) + 2*loss_adv_foreground


        # todo: self attn loss
        # pred_target_fake_label = pred_target_fake.permute(0, 2, 3, 1)
        # _, pred_target_fake_label = torch.max(pred_target_fake_label, -1)
        # bce_loss = nn.BCEWithLogitsLoss()
        # attn_loss = bce_loss(pred_target_edge,
        #                      self.get_foreground_attention(
        #                          pred_target_fake_label.view(pred_target_fake_label.shape[0], -1,
        #                                                      pred_target_fake_label.shape[1],
        #                                                      pred_target_fake_label.shape[2]).cuda()))

        # in target domain compute edge loss - weakly constraint
        # edge_loss = self._compute_edge_loss(pred_target_edge, self.channel_to_label(F.softmax(pred_target_fake)))
        loss = self.lambda_adv_target * loss
        # loss = self.lambda_adv_target * adv_loss + 0.1*self.lambda_adv_target * edge_loss
        # loss = self.lambda_adv_target * adv_loss + 0.1*self.lambda_adv_target * attn_loss
        loss.backward()

        # update loss
        self.optimizer_G.step()

        # save image for discriminator use
        self.target_image = pred_target_fake.detach()
        # self.pred_fake_edge = F.sigmoid(pred_target_edge).detach()

        # self.pred_fake_edge = pred_target_edge.detach()
        self.target_input_image = images.detach()

        # record log
        self.loss_target_foreground_value += loss_adv_foreground.data.cpu().numpy()
        self.loss_target_value += loss_adv.data.cpu().numpy()

        self.loss_dict['Global_GAN_adv'] += loss_adv.data.cpu().numpy()
        self.loss_dict['Foreground_GAN_adv'] += loss_adv_foreground.data.cpu().numpy()
        del loss, loss_adv, loss_adv_foreground

    def dis_update(self, labels=None):
        """
                use [gen_source_update / gen_target_update]'s image to discriminator,
                so you  don' t need to give any parameter
                """
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        self.optimizer_D_foreground.zero_grad()

        # Enable D backpropgation, train D
        for param in self.model_D.parameters():
            param.requires_grad = True

        for param in self.model_D_foreground.parameters():
            param.requires_grad = True

        # we don't train target's G weight, we only train source'G
        self.target_image = self.target_image.detach()
        # compute adv loss function

        # cobime predict and use predict output get edge
        # net_input = torch.cat((F.softmax(self.source_image), self.pred_get_edges(self.source_image)), dim=1)
        # net_input = torch.cat((F.softmax(self.source_image), self.pred_real_edge), dim=1)
        net_input = F.softmax(self.source_image, dim=1)

        # d_out_real, _ = self.model_D(F.softmax(self.source_image), label=self.source_input_image)
        # d_out_real, self.pred_real_d_proj = self.model_D(net_input, label=None)
        # d_out_real, _ = self.model_D(net_input, label=self.pred_real_edge)
        d_out_real, _ = self.model_D(net_input, label=None)
        d_out_foreground_real, _ = self.model_D_foreground(net_input, label=self.pred_real_edge)
        # d_out_foreground_real, _ = self.model_D(net_input, label=self.pred_real_edge)

        # d_out_real, self.pred_real_d_proj = self.model_D(net_input, label=self.pred_source_edge_mini)
        # d_out_real, self.pred_real_d_proj = self.model_D(net_input, label=self.pred_source_edge_mini-0.5)
        # d_out_real, self.pred_real_d_proj = self.model_D(net_input, label=(self.pred_source_edge_mini - 0.5)*2)
        # loss_real = self._compute_adv_loss_real(d_out_real, self.adv_loss_opt)
        # loss_real /= 2
        # loss_real.backward()

        # cobime predict and use predict output get edge
        # net_input = torch.cat((F.softmax(self.target_image), self.pred_fake_edge), dim=1)
        net_input = F.softmax(self.target_image, dim=1)
        # d_out_fake, _ = self.model_D(F.softmax(self.target_image), label=self.target_input_image)
        # d_out_fake, self.pred_fake_d_proj = self.model_D(net_input, label=None)
        # d_out_fake, _ = self.model_D(net_input, label=self.pred_fake_edge)
        d_out_fake, _ = self.model_D(net_input, label=None)
        d_out_foreground_fake, _ = self.model_D_foreground(net_input, label=self.pred_fake_edge)

        # d_out_fake, self.pred_fake_d_proj = self.model_D(net_input, label=self.pred_target_edge_mini)
        # d_out_fake, self.pred_fake_d_proj = self.model_D(net_input, label=self.pred_target_edge_mini-0.5)
        # d_out_fake = self.model_D(F.softmax(self.target_image), label=self.interp_mini(self.target_image_input_image))

        # loss_fake = self._compute_adv_loss_fake(d_out_fake, self.adv_loss_opt)
        # loss_fake /= 2

        # loss = loss_real + loss_fake
        # loss = self.loss_hinge_dis(d_out_fake, d_out_real)
        # foreground part
        if self.adv_loss_opt == "hinge":
            loss = self.loss_hinge_dis(d_out_foreground_fake, d_out_foreground_real)
            # print("fore loss=", loss)
            # print("hinge")
        else:
            # print("self.adv_loss_opt")
            loss_fake = self._compute_adv_loss_fake(d_out_foreground_fake, self.adv_loss_opt)
            loss_real = self._compute_adv_loss_real(d_out_foreground_real, self.adv_loss_opt)
            loss = (loss_real + loss_fake)/2
            del loss_fake, loss_real
        loss.backward()
        # self.loss_d_foreground_value += loss.data.cpu().numpy()
        self.loss_dict['Foreground_GAN_dis'] += loss.data.cpu().numpy()

        # global part
        if self.adv_loss_opt == "hinge":
            loss = self.loss_hinge_dis(d_out_fake, d_out_real)
            # print("fore loss=", loss)
        else:
            loss_fake = self._compute_adv_loss_fake(d_out_fake, self.adv_loss_opt)
            loss_real = self._compute_adv_loss_real(d_out_real, self.adv_loss_opt)
            loss = (loss_real + loss_fake)/2
            del loss_fake, loss_real
        loss.backward()
        # self.loss_d_value += loss.data.cpu().numpy()
        self.loss_dict['Global_GAN_dis'] += loss.data.cpu().numpy()

        # update loss
        self.optimizer_D.step()
        self.optimizer_D_foreground.step()
        del loss
        # self.optimizer_Attn.step()

        # record log
        # self.loss_d_value += loss_real.data.cpu().numpy() + loss_fake.data.cpu().numpy()


    def loss_hinge_dis(self, dis_fake, dis_real):
        loss = torch.mean(F.relu(1. - dis_real))
        loss += torch.mean(F.relu(1. + dis_fake))
        return loss

    def loss_hinge_gen(self, g_fake):
        loss = -torch.mean(g_fake)
        return loss

    def show_each_loss(self):
        # print(
        #     "trainer - iter = {0:8d}/{1:8d}, loss_G_source_1 = {2:.3f} loss_G_adv1 = {3:.5f} loss_G_adv_foreground = {4:.5f}  loss_D1 = {5:.3f} loss_D1_foreground = {6:.3f}".format(
        #         self.i_iter, self.num_steps, self.loss_source_value, float(self.loss_target_value),
        #         float(self.loss_target_foreground_value), float(self.loss_d_value), float(self.loss_d_foreground_value)))
        message = '(epoch: %d, iters: %d) ' % (self.num_steps, self.i_iter)
        for k, v in self.loss_dict.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def pred_get_edges(self, t):
        pred = F.softmax(t, dim=1)
        # print("predl  origin softmax shape", pred.shape)
        pred = pred.cpu().data.numpy()
        # print("predl softmax shape", pred.shape)
        pred = pred.transpose(0, 2, 3, 1)
        pred_label = np.asarray(np.argmax(pred, axis=3), dtype=np.uint8)
        # print("pred_label shape", pred_label.shape)
        # print("pred_label size", pred_label.size())
        pred_label = pred_label[:, np.newaxis, :, :]
        # print("pred_label newaxis shape", pred_label.shape)

        pred_edge = self.label_get_edges(torch.tensor(pred_label).cuda()).detach()
        return pred_edge

    def channel_to_label(self, pred):
        pred = pred.permute(0, 2, 3, 1)
        _, output = torch.max(pred, -1)
        return output

    def get_foreground_attention_groups(self, label):
        foreground_groups = []
        # foreground_map = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 255]
        foreground_groups_dict = {
            "riders": [12, 17, 18],
            "cars": [13, 14, 15, 16],
            "sidewalk": [1],
            "person": [11],
            "light": [6],
            "sign": [7]
        }
        for group_name, group_list in foreground_groups_dict.items():
            foreground_attn = torch.cuda.ByteTensor(label.size()).zero_()
            for item in group_list:
                # print(value)
                foreground_attn[label == item] = 1
            foreground_groups += [foreground_attn]

        return torch.cat(foreground_groups, dim=1)

    def get_foreground_attention(self, label):
        # foreground_attn = torch.cuda.ByteTensor(label.size()).zero_()
        foreground_attn = torch.cuda.ByteTensor(label.size()).zero_()
        # choose which label be weakly supervised label
        # 0 > road: 87.89
        # 1 > sidewalk: 31.54
        # 2 > building: 77.69
        # 3 > wall: 23.56
        # 4 > fence: 20.1
        # 5 > pole: 25.49
        # 6 > light: 27.14
        # 7 > sign: 26.57
        # 8 > vegetation: 70.86
        # 9 > terrain: 24.19
        # 10 > sky: 71.61
        # 11 > person: 55.79
        # 12 > rider: 21.21
        # 13 > car: 82.2
        # 14 > truck: 31.67
        # 15 > bus: 44.69
        # 16 > train: 0.9
        # 17 > motocycle: 22.13
        # 18 > bicycle: 25.14

        # ignore background label include 255-ignore label
        # foreground_map = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 255]
        # foreground_map = [11, 12, 13, 14, 15, 16, 17, 18]
        # todo:change foreground classes
        # foreground_map = [12, 13, 14, 15, 16, 17, 18]

        # foreground_map = [11, 12, 13, 14, 15, 16, 17, 18]

        # foreground_map = [11, 12, 13, 14, 15, 16, 17, 18]

        # foreground_map = [1, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
        foreground_map = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
        # foreground_map = [6, 7, 11, 12, 13, 14, 15, 16, 17, 18]

        # choose which label be weakly supervised label

        for value in foreground_map:
            # print(value)
            foreground_attn[label == value] = 1
        self.saliency_mask = foreground_attn.detach().float()

        return foreground_attn.float()

    def label_get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        # t = t.view(1, 1, t.shape[0], t.shape[1])
        # print("edge shape", edge.shape)
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def _compute_adv_loss_real(self, d_out_real, loss_opt="bce", label=0):
        """
                compute adversarial loss function, can choose loss opt
                :param d_out_fake:
                :param loss_opt:  [wgan-gp / hinge / bce]
                :param label:
                :return:
                """
        assert loss_opt == "bce" or loss_opt == "wgan-gp"
        # print("_compute_adv_loss_real loss_opt", loss_opt)
        d_loss_real = None
        if loss_opt == 'wgan-gp':
            d_loss_real = - d_out_real.mean()
        elif loss_opt == 'bce':
            bce_loss = torch.nn.BCEWithLogitsLoss()
            d_loss_real = bce_loss(d_out_real,
                                   Variable(torch.FloatTensor(d_out_real.data.size()).fill_(label)).cuda(
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
        # print("_compute_adv_loss_fake loss_opt", loss_opt)
        d_loss_fake = None
        if loss_opt == 'wgan-gp':
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

    # todo: this part will modify
    def _compute_edge_loss(self, seg, labels):
        # in source domain compute edge loss
        # Todo : here use softmax maybe wrong
        edge = self.label_get_edges(labels.view(labels.shape[0], 1, labels.shape[1], labels.shape[2]).cuda())
        foreground_mask = self.get_foreground_attention(
            labels.view(labels.shape[0], 1, labels.shape[1], labels.shape[2]).cuda())
        # ignore foreground part
        foreground_mask[foreground_mask == 1] = 255
        # but we need edge info no matter it is bg or fg
        foreground_mask[edge == 1] = 1

        bce_loss = nn.BCEWithLogitsLoss()

        positions = foreground_mask.contiguous().view(-1) < 255.0
        loss = bce_loss(seg.contiguous().view(-1)[positions],
                        foreground_mask.contiguous().view(-1)[positions])

        foreground_mask[foreground_mask == 255] = 0.5
        self.saliency_mask = foreground_mask

        return loss

    # def _resize(self, img, size=None):
    #     # resize to source size
    #     interp = nn.Upsample(size=size, align_corners=False, mode='bilinear')
    #     img = interp(img)
    #     return img

    def _lr_poly(self, base_lr, i_iter, max_iter, power):
        return base_lr * ((1 - float(i_iter) / max_iter) ** power)

    def _adjust_learning_rate_D(self, optimizer, i_iter):
        lr = self._lr_poly(self.lr_d, i_iter, self.num_steps, self.decay_power)
        for i, group in enumerate(optimizer.param_groups):
            optimizer.param_groups[i]['lr'] = lr

    def _adjust_learning_rate_G(self, optimizer, i_iter):
        lr = self._lr_poly(self.lr_g, i_iter, self.num_steps, self.decay_power)
        for i, group in enumerate(optimizer.param_groups):
            optimizer.param_groups[i]['lr'] = lr
        # print("len(optimizer.param_groups)", len(optimizer.param_groups))
        # optimizer.param_groups[0]['lr'] = lr
        # if len(optimizer.param_groups) > 1:
        #     optimizer.param_groups[1]['lr'] = lr * 10

    def init_each_epoch(self, i_iter):
        self.i_iter = i_iter
        self.loss_dict = {k: 0 for k in self.loss_names}
        # self.loss_d_value = 0
        # self.loss_d_foreground_value = 0
        #
        # self.loss_source_value = 0
        # self.loss_target_value = 0
        # self.loss_edge_value = 0


        # self.loss_target_foreground_value = 0

        # self.loss_target_weakly_seg_value = 0

        # self.loss_d_attn_value = 0

    def update_learning_rate(self):
        if self.optimizer_G:
            self.optimizer_G.zero_grad()
            self._adjust_learning_rate_G(self.optimizer_G, self.i_iter)

        if self.optimizer_D:
            self.optimizer_D.zero_grad()
            self._adjust_learning_rate_D(self.optimizer_D, self.i_iter)

        if self.optimizer_D_foreground:
            self.optimizer_D_foreground.zero_grad()
            self._adjust_learning_rate_D(self.optimizer_D_foreground, self.i_iter)

    @property
    def discriminator_gamma(self):
        # return str(self.model_D.gamma.float())
        return 0

    def snapshot_image_save(self, dir_name="check_output/", src_save=True, target_save=True):
        """
                check model training status,
                will output image to config["image_save_dir"]
                """
        if not os.path.exists(os.path.join(dir_name, "Image_source_domain_seg")):
            os.makedirs(os.path.join(dir_name, "Image_source_domain_seg"))
        if not os.path.exists(os.path.join(dir_name, "Image_target_domain_seg")):
            os.makedirs(os.path.join(dir_name, "Image_target_domain_seg"))

        if src_save:
            # save image
            dataset_name = self.hyperparameters["source_dataset"]
            input_image_name = os.path.join("data", dataset_name, "images", self.source_label_path[0])
            save_name = os.path.join(dir_name, "Image_source_domain_seg", '%s_input.png' % self.i_iter)
            shutil.copyfile(input_image_name, save_name)

            # save label
            if dataset_name == "SYNTHIA":
                label_name = os.path.join("data", dataset_name, "labels-rgb", self.source_label_path[0])
            else:
                label_name = os.path.join("data", dataset_name, "labels", self.source_label_path[0])
            save_name = os.path.join(dir_name, "Image_source_domain_seg", '%s_label.png' % self.i_iter)
            shutil.copyfile(label_name, save_name)

            # save output image
            paint_predict_image(self.source_image).save('check_output/Image_source_domain_seg/%s.png' % self.i_iter)

        if target_save:
            target_name = os.path.join("data", "Cityscapes", "data", "leftImg8bit", "train", self.target_image_path[0])
            save_name = os.path.join(dir_name, "Image_target_domain_seg", '%s_input.png' % self.i_iter)
            shutil.copyfile(target_name, save_name)
            paint_predict_image(self.target_image).save('check_output/Image_target_domain_seg/%s.png' % self.i_iter)

    def snapshot_edge_save(self, dir_name="check_output/", src_save=True, target_save=True, labels=None):
        """
                check model training status,
                will output image to config["image_save_dir"]
                """
        if not os.path.exists(os.path.join(dir_name, "Image_source_domain_seg")):
            os.makedirs(os.path.join(dir_name, "Image_source_domain_seg"))
        if not os.path.exists(os.path.join(dir_name, "Image_target_domain_seg")):
            os.makedirs(os.path.join(dir_name, "Image_target_domain_seg"))

        # save GT edge
        # if no use is not None will be ambiguous
        # if labels is not None:
        #     labels = labels.view(labels.shape[0], 1, labels.shape[1], labels.shape[2])
        #     self.tensor_to_PIL(self.label_get_edges(labels.cuda())).save(
        #         'check_output/Image_source_domain_seg/%s_edge_label.png' % self.i_iter)
            # self.tensor_to_PIL(self.get_foreground_attention(labels.cuda())).save('check_output/Image_source_domain_seg/%s_edge_label.png' % self.i_iter)

        # save output image
        if src_save:
            # record edge attn
            self.tensor_to_PIL(self.saliency_mask).save(
                'check_output/Image_source_domain_seg/%s_compute_attn.png' % self.i_iter)

            self.tensor_to_PIL(self.pred_real_edge).save(
                'check_output/Image_source_domain_seg/%s_edge.png' % self.i_iter)
            # interp = nn.Upsample(size=self.input_size, align_corners=True, mode='bilinear')
            # self.pred_real_d_proj = interp(nn.Sigmoid()(self.pred_real_d_proj))

            # self.pred_real_d_proj = interp(self.pred_real_d_proj)
            # self.tensor_to_PIL(self.pred_real_d_proj).save('check_output/Image_source_domain_seg/%s_proj.png' % self.i_iter)

            # pred_real_d_proj_sig = nn.Sigmoid()(self.pred_real_d_proj)
            # self.tensor_to_PIL(pred_real_d_proj_sig).save(
            #     'check_output/Image_source_domain_seg/%s_proj_sig.png' % self.i_iter)
            # pred_real_d_proj_softmax = nn.Softmax()(self.pred_real_d_proj)
            # self.tensor_to_PIL(pred_real_d_proj_softmax).save(
            #     'check_output/Image_source_domain_seg/%s_proj_softmax.png' % self.i_iter)
            # self.tensor_to_PIL((self.pred_real_edge-0.25)*2).save('check_output/Image_source_domain_seg/%s_edge_light.png' % self.i_iter)

            # print("pred_real_edge max =", torch.max(self.pred_real_edge).cpu().numpy())
            # print("pred_real_edge mean =", torch.mean(self.pred_real_edge).cpu().numpy())

        if target_save:
            self.tensor_to_PIL(self.pred_fake_edge).save(
                'check_output/Image_target_domain_seg/%s_edge.png' % self.i_iter)
            # self.tensor_to_PIL((self.pred_fake_edge-0.25)*2).save('check_output/Image_target_domain_seg/%s_edge_light.png' % self.i_iter)
            # print("pred_fake_edge max =", torch.max(self.pred_fake_edge).cpu().numpy())
            # print("pred_fake_edge mean =", torch.mean(self.pred_fake_edge).cpu().numpy())
            # print("self.pred_fake_d_proj) max = ", torch.max(self.pred_fake_d_proj))
            # interp_target = nn.Upsample(size=self.input_size_target, align_corners=False, mode='bilinear')
            # self.pred_fake_d_proj = interp_target(nn.Sigmoid()(self.pred_fake_d_proj))

            # self.pred_fake_d_proj = interp_target((self.pred_fake_d_proj))
            # self.tensor_to_PIL(self.pred_fake_d_proj).save('check_output/Image_target_domain_seg/%s_proj.png' % self.i_iter)

            # pred_fake_d_proj_sig = nn.Sigmoid()(self.pred_fake_d_proj)
            # self.tensor_to_PIL(pred_fake_d_proj_sig).save(
            #     'check_output/Image_target_domain_seg/%s_proj_sig.png' % self.i_iter)
            # pred_fake_d_proj_softmax = nn.Softmax()(self.pred_fake_d_proj)
            # self.tensor_to_PIL(pred_fake_d_proj_softmax).save(
            #     'check_output/Image_target_domain_seg/%s_proj_softmax.png' % self.i_iter)

    def save_model(self, snapshot_save_dir="./model_save"):
        """
                save model to .pth file
                will output model to config["snapshot_save_dir"]
                """
        print('taking pth in shapshot dir ...')
        torch.save(self.model.state_dict(), os.path.join(snapshot_save_dir, 'GTA5_' + str(self.i_iter) + '.pth'))
        torch.save(self.model_D.state_dict(), os.path.join(snapshot_save_dir, 'GTA5_' + str(self.i_iter) + '_D1.pth'))
        torch.save(self.model_D_foreground.state_dict(),
                   os.path.join(snapshot_save_dir, 'GTA5_' + str(self.i_iter) + '_D_foreground.pth'))

    def restore(self, model_name=None, num_classes=19, restore_from=None):

        # self.model = Res_Deeplab(num_classes=num_classes)
        print("check restore from", restore_from)
        if restore_from[:4] == 'http':
            saved_state_dict = model_zoo.load_url(restore_from)
            new_params = self.model.state_dict().copy()
            # print("new_params", new_params)
            # if num_classes != 19:
            #     # print("saved_state_dict =", saved_state_dict)
            #     for name, param in saved_state_dict.items():
            #         name_parts = name.split('.')
            #         name = '.'.join(name_parts[1:])
            #         if name not in new_params:
            #             continue
            #         print("replace")
            #         param = param.data
            #         new_params[name].copy_(param)
            #     print("before model load")
            #     print("self.model.state_dict()")
            #     print(str(self.model.state_dict())[:100])
            #     self.model.load_state_dict(new_params)
            #     print("after model load")
            #     print("self.model.state_dict()")
            #     print(str(self.model.state_dict())[:100])
            # else:
            for i in saved_state_dict:
                # Scale.layer5.conv2d_list.3.weight
                i_parts = i.split('.')
                # if not num_classes == 19 or not i_parts[1] == 'layer5':
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            # new_params = saved_state_dict
            print("before model load")
            print("self.model.state_dict()")
            print(str(self.model.state_dict())[:100])
            self.model.load_state_dict(new_params)
            del new_params
            print("after model load")
            print("self.model.state_dict()")
            print(str(self.model.state_dict())[:100])

        else:
            print("use own pre-trained")
            print("before model load")
            print("self.model.state_dict()")
            print(str(self.model.state_dict())[:100])
            saved_state_dict = torch.load(restore_from)
            self.model.load_state_dict(saved_state_dict)
            print("after model load")
            print("self.model.state_dict()")
            print(str(self.model.state_dict())[:100])
        self.init_opt()

    def restore_D(self, restore_from="./GTA5_45000_D1.pth"):

        print("use own D pre-trained")
        saved_state_dict = torch.load(restore_from)
        self.model_D.load_state_dict(saved_state_dict)

        self.init_opt()

    # todo: move to util.py
    def tensor_to_PIL(self, tensor):
        pilTrans = transforms.ToPILImage()
        image = pilTrans(tensor.cpu().squeeze(0))
        return image


def paint_predict_image(predict_image):
    """input model's output image it will paint color """
    # ===============for colorize mask==============
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    def colorize_mask(mask):
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(palette)

        return new_mask

    def output_to_image(output):
        # input
        # ------------------
        #   G's output feature map :(c, w, h, num_classes)
        #
        #
        # output
        # ------------------
        #   output_color : PIL Image paint segmentaion color (1024, 2048)
        #
        #
        interp = nn.Upsample(size=(1024, 2048), align_corners=False, mode='bilinear')
        output = interp(output).permute(0, 2, 3, 1)
        _, output = torch.max(output, -1)
        output = output.cpu().data[0].numpy().astype(np.uint8)
        output_color = colorize_mask(output)

        return output_color

    return output_to_image(predict_image)
