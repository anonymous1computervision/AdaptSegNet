import os
import shutil
import pdb

# from tensorboard_logger import configure, log_value
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils import model_zoo

import numpy as np
from PIL import Image

# from model.deeplab_multi import Res_Deeplab
from model.deeplab_single_attention import Res_Deeplab
# from model.discriminator import FCDiscriminator
from model.xiao_discriminator import XiaoDiscriminator
from model.xiao_attention import XiaoAttention
from model.xiao_discriminator_addition_attention import XiaoAttentionDiscriminator
from model.xiao_pretrained_attention_discriminator import XiaoPretrainAttentionDiscriminator

from utils.loss import CrossEntropy2d

class AdaptSeg_Attn_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(AdaptSeg_Attn_Trainer, self).__init__()
        self.hyperparameters = hyperparameters
        # input size setting
        self.input_size = (hyperparameters["input_size_h"], hyperparameters["input_size_w"])
        self.input_size_target = (hyperparameters["input_target_size_h"], hyperparameters["input_target_size_w"])

        # training setting
        self.num_steps = hyperparameters["num_steps"]

        # log setting
        # configure(hyperparameters['config_path'])

        # cuda setting
        self.gpu = hyperparameters['gpu']
        cudnn.benchmark = True

        # init G
        if hyperparameters["model"] == 'DeepLab':
            self.model = Res_Deeplab(num_classes=hyperparameters["num_classes"])
        self.model_attn = XiaoAttention(hyperparameters["num_classes"])
        # init D
        # self.model_D = FCDiscriminator(num_classes=hyperparameters['num_classes'])
        self.model_D = XiaoAttentionDiscriminator(num_classes=hyperparameters['num_classes'])
        # self.model_D = XiaoPretrainAttentionDiscriminator(num_classes=hyperparameters['num_classes'])

        self.model.train()
        self.model.cuda(self.gpu)
        # self.model_attn.train()
        # self.model_attn.cuda(self.gpu)
        self.model_D.train()
        self.model_D.cuda(self.gpu)

        # for dynamic adjust lr setting
        self.decay_power = hyperparameters['decay_power']

        # init optimizer
        self.lr_g = hyperparameters['lr_g']
        self.lr_d = hyperparameters['lr_d']
        self.lr_attn = hyperparameters['lr_attn']
        self.momentum = hyperparameters['momentum']
        self.weight_decay = hyperparameters['weight_decay']
        self.beta1 = hyperparameters['beta1']
        self.beta2 = hyperparameters['beta2']
        self.init_opt()

        # for [log / check output]
        self.loss_d_value = 0
        self.loss_source_value = 0
        self.loss_target_value = 0
        self.i_iter = 0
        self.source_label_path = None
        self.target_image_path = None

        # for generator
        self.lambda_seg = hyperparameters['gen']['lambda_seg']
        self.lambda_adv_target = hyperparameters['gen']['lambda_adv_target']
        self.lambda_g_attention = hyperparameters['gen']['lambda_attn']
        self.decay_power = hyperparameters['decay_power']

        # for discriminator
        self.adv_loss_opt = hyperparameters['dis']['adv_loss_opt']
        self.lambda_attn = hyperparameters['dis']['lambda_attn']

        self.source_image = None
        self.target_image = None
        self.inter_mini = nn.Upsample(size=self.input_size_target, align_corners=False,
                                    mode='bilinear')
    def init_opt(self):
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

        # self.optimizer_Attn = optim.Adam([p for p in self.model_attn.parameters() if p.requires_grad],
        #                               lr=self.lr_attn, betas=(self.beta1, self.beta2))
        # self.optimizer_Attn.zero_grad()
        # self._adjust_learning_rate_D(self.optimizer_Attn, 0)

    def forward(self, images):
        self.eval()
        predict_seg = self.model(images)
        self.train()
        return predict_seg

    def _resize(self, img, size=None):
        # resize to source size
        interp = nn.Upsample(size=size, align_corners=False, mode='bilinear')
        img = interp(img)
        return img

    def gen_source_update(self, images, labels, label_path=None):
        """
                Input source domain image and compute segmentation loss.

                :param images:
                :param labels:
                :param label_path: just for save path to record model predict, use in  snapshot_image_save function

                :return:
                """
        self.optimizer_G.zero_grad()
        # self.optimizer_Attn.zero_grad()
        self.optimizer_D.zero_grad()

        # Disable D backpropgation, we only train G
        for param in self.model_D.parameters():
            param.requires_grad = False
        for param in self.model_attn.parameters():
            param.requires_grad = True

        self.source_label_path = label_path

        # get predict output
        pred_source_real, attn = self.model(images)
        # print("conv last 2 shape", conv_last_2.shape)
        # interp = nn.Upsample(size=self.input_size, align_corners=False, mode='bilinear')
        # pred_source_real = interp(pred_source_real)
        pred_source_real = self._resize(pred_source_real, size=self.input_size)
        # in source domain compute segmentation loss
        seg_loss = self._compute_seg_loss(pred_source_real, labels)
        seg_loss = self.lambda_seg * seg_loss
        # seg_loss.backward(retain_graph=True)

        # pred_source_attn_real = self.model_attn(conv_last_2)
        # attention use low resolution
        pred_source_attn_real = self._resize(attn, size=self.input_size)
        # in source domain compute attention loss
        attn_loss = self._compute_seg_loss(pred_source_attn_real, labels)
        # proper normalization
        attn_loss = self.lambda_g_attention * attn_loss
        loss = seg_loss + attn_loss
        # attn_loss.backward()
        loss.backward()

        # update loss
        self.optimizer_G.step()
        # self.optimizer_Attn.step()

        # save image for discriminator use
        self.source_image = pred_source_real.detach()

        # record log
        self.loss_source_value += seg_loss.data.cpu().numpy()

    def gen_target_update(self, images, image_path):
        """
                 Input target domain image and compute adversarial loss.

                :param images:
                :param image_path: just for save path to record model predict, use in  snapshot_image_save function
                :return:
                """
        self.optimizer_G.zero_grad()
        # self.optimizer_Attn.zero_grad()
        self.optimizer_D.zero_grad()

        # Disable D backpropgation, we only train G
        for param in self.model_D.parameters():
            param.requires_grad = False
        for param in self.model_attn.parameters():
            param.requires_grad = False

        self.target_image_path = image_path

        # get predict output
        pred_target_fake, _ = self.model(images)

        # resize to target size
        pred_target_fake = self._resize(pred_target_fake, size=self.input_size_target)

        # d_out_fake = model_D(F.softmax(pred_target_fake), inter_mini(F.softmax(pred_target_fake)))
        # d_out_fake, _ = self.model_D(F.softmax(pred_target_fake), model_attn=self.model_attn)
        d_out_fake, _ = self.model_D(F.softmax(pred_target_fake))

        # compute loss function
        # wants to fool discriminator
        adv_loss = self._compute_adv_loss_real(d_out_fake, loss_opt=self.adv_loss_opt)
        loss = self.lambda_adv_target * adv_loss
        loss.backward()

        # update loss
        self.optimizer_G.step()

        # save image for discriminator use
        self.target_image = pred_target_fake.detach()

        # record log
        self.loss_target_value += adv_loss.data.cpu().numpy()

    def dis_update(self, labels=None):
        """
                use [gen_source_update / gen_target_update]'s image to discriminator,
                so you  don' t need to give any parameter
                """
        self.optimizer_G.zero_grad()
        # self.optimizer_Attn.zero_grad()
        self.optimizer_D.zero_grad()

        # Enable D backpropgation, train D
        for param in self.model_D.parameters():
            param.requires_grad = True
        for param in self.model_attn.parameters():
            # param.requires_grad = True
            param.requires_grad = False
        # we don't train target's G weight, we only train source'G
        self.target_image = self.target_image.detach()
        # compute adv loss function
        # d_out_real, _ = self.model_D(F.softmax(self.source_image), label=None, model_attn=self.model_attn)
        d_out_real, _ = self.model_D(F.softmax(self.source_image), label=None)
        # Todo:for memory issue divide 2 bp
        loss_real = self._compute_adv_loss_real(d_out_real, self.adv_loss_opt)
        loss_real /= 2
        self.loss_d_value += loss_real.data.cpu().numpy()
        loss_real.backward()
        self.optimizer_D.step()


        _, attn = self.model_D(F.softmax(self.source_image), label=None)
        # attention resize to source size
        d_attn = self._resize(attn, size=self.input_size)

        # in source domain compute attention loss
        loss_attn = self.lambda_attn * self._compute_seg_loss(d_attn, labels)
        self.loss_d_value += loss_attn.data.cpu().numpy()
        # loss = loss_real + loss_attn
        loss_attn.backward()
        self.optimizer_D.step()

        # d_out_fake, _ = self.model_D(F.softmax(self.target_image), label=None, model_attn=self.model_attn)
        d_out_fake, _ = self.model_D(F.softmax(self.target_image), label=None)
        loss_fake = self._compute_adv_loss_fake(d_out_fake, self.adv_loss_opt)
        loss_fake /= 2
        self.loss_d_value += loss_fake.data.cpu().numpy()
        loss_fake.backward()

        # compute attn loss function
        # interp = nn.Upsample(size=self.input_size, align_corners=False, mode='bilinear')
        # loss_attn = self._compute_seg_loss(interp(attn), labels)

        # compute total loss function
        # loss = loss_real + loss_fake + self.lambda_attn*loss_attn
        # loss = loss_real + loss_fake + loss_attn
        # loss.backward()

        # update loss
        self.optimizer_D.step()
        # self.optimizer_Attn.step()

        # record log
        # self.loss_d_value += loss_real.data.cpu().numpy() + loss_fake.data.cpu().numpy()

    def show_each_loss(self):
        print("attn trainer - iter = {0:8d}/{1:8d}, loss_G_source_1 = {2:.3f} loss_G_adv1 = {3:.3f} loss_D1 = {4:.3f}".format(
            self.i_iter, self.num_steps, self.loss_source_value, float(self.loss_target_value), float(self.loss_d_value)))

    def _compute_adv_loss_real(self, d_out_real, loss_opt="bce", label=0):
        """
                compute adversarial loss function, can choose loss opt
                :param d_out_fake:
                :param loss_opt:  [wgan-gp / hinge / bce]
                :param label:
                :return:
                """
        d_loss_real = None
        if loss_opt == 'wgan-gp':
            d_loss_real = - d_out_real.mean()
        elif loss_opt == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
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
        d_loss_fake = None
        if loss_opt == 'wgan-gp':
            d_loss_fake = - d_out_fake.mean()
        elif loss_opt == 'hinge':
            # d_loss_fake = - d_out_fake.mean()
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
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

    def _adjust_learning_rate_Attn(self, optimizer, i_iter):
        lr = self._lr_poly(self.lr_attn, i_iter, self.num_steps, self.decay_power)
        for i, group in enumerate(optimizer.param_groups):
            optimizer.param_groups[i]['lr'] = lr
    def init_each_epoch(self, i_iter):
        self.i_iter = i_iter
        self.loss_d_value = 0
        self.loss_source_value = 0
        self.loss_target_value = 0
        self.source_image = None
        self.target_image = None

    def update_learning_rate(self):
        if self.optimizer_G:
            self.optimizer_G.zero_grad()
            self._adjust_learning_rate_G(self.optimizer_G, self.i_iter)

        if self.optimizer_D:
            self.optimizer_D.zero_grad()
            self._adjust_learning_rate_D(self.optimizer_D, self.i_iter)

        # if self.optimizer_Attn:
        #     self.optimizer_Attn.zero_grad()
        #     self._adjust_learning_rate_Attn(self.optimizer_Attn, self.i_iter)

    def snapshot_image_save(self, dir_name="check_output/"):
        """
                check model training status,
                will output image to config["image_save_dir"]
                """
        if not os.path.exists(os.path.exists(os.path.join(dir_name, "Image_source_domain_seg"))):
            os.makedirs(os.path.join(dir_name, "Image_source_domain_seg"))
        if not os.path.exists(os.path.exists(os.path.join(dir_name, "Image_target_domain_seg"))):
            os.makedirs(os.path.join(dir_name, "Image_target_domain_seg"))

        # save label
        label_name = os.path.join("data", "GTA5", "labels", self.source_label_path[0])
        save_name = os.path.join(dir_name, "Image_source_domain_seg", '%s_label.png' % self.i_iter)
        shutil.copyfile(label_name, save_name)

        target_name = os.path.join("data", "Cityscapes", "data", "leftImg8bit", "train", self.target_image_path[0])
        save_name = os.path.join(dir_name, "Image_target_domain_seg", '%s_label.png' % self.i_iter)
        shutil.copyfile(target_name, save_name)

        # save output image
        paint_predict_image(self.source_image).save('check_output/Image_source_domain_seg/%s.png' % self.i_iter)
        paint_predict_image(self.target_image).save('check_output/Image_target_domain_seg/%s.png' % self.i_iter)

    def save_model(self, snapshot_save_dir="./model_save"):
        """
                save model to .pth file
                will output model to config["snapshot_save_dir"]
                """
        print('taking pth in shapshot dir ...')
        torch.save(self.model.state_dict(), os.path.join(snapshot_save_dir, 'GTA5_' + str(self.i_iter) + '.pth'))
        torch.save(self.model_D.state_dict(), os.path.join(snapshot_save_dir, 'GTA5_' + str(self.i_iter) + '_D1.pth'))

    def restore(self, model_name=None, num_classes=19, restore_from=None):
        if model_name == 'DeepLab':
            # self.model = Res_Deeplab(num_classes=num_classes)
            print("check restore from", restore_from)
            if restore_from[:4] == 'http':
                saved_state_dict = model_zoo.load_url(restore_from)
                new_params = self.model.state_dict().copy()
                for i in saved_state_dict:
                    # Scale.layer5.conv2d_list.3.weight
                    i_parts = i.split('.')
                    if not num_classes == 19 or not i_parts[1] == 'layer5':
                        new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # new_params = saved_state_dict
                self.model.load_state_dict(new_params)
            else:
                print("use own pre-trained")
                saved_state_dict = torch.load(restore_from)
                self.model.load_state_dict(saved_state_dict)

            self.init_opt()



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
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear')
        output = interp(output).permute(0, 2, 3, 1)
        _, output = torch.max(output, -1)
        output = output.cpu().data[0].numpy().astype(np.uint8)
        output_color = colorize_mask(output)

        return output_color
    return output_to_image(predict_image)

