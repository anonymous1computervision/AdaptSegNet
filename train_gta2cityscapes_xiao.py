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

# from model.deeplab_multi import Res_Deeplab
from model.deeplab_single import Res_Deeplab
from model.discriminator import FCDiscriminator
from model.xiao_discriminator import XiaoDiscriminator
from model.xiao_attention_discriminator import XiaoAttentionDiscriminator

from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './data/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
# INPUT_SIZE = '512,256'
# INPUT_SIZE = '320,180'
DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
# INPUT_SIZE_TARGET = '512,256'
# INPUT_SIZE_TARGET = '256,128'

LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 100000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
# RESTORE_FROM = '.\snapshots\single_DA_baseline\GTA5_50000.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4

LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

TARGET = 'cityscapes'
SET = 'train'

DEBUG_MODE = False

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")

    return parser.parse_args()


args = get_arguments()

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
    output = interp(output).permute(0,2,3, 1)
    _, output = torch.max(output, -1)
    output = output.cpu().data[0].numpy().astype(np.uint8)
    output_color = colorize_mask(output)

    return output_color


# ===============for model==============
def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    # ASPP layer learning rate *10
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
    #     # optimizer.param_groups[1]['lr'] = lr * 10
        optimizer.param_groups[1]['lr'] = lr


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
    #     # optimizer.param_groups[1]['lr'] = lr * 10
        optimizer.param_groups[1]['lr'] = lr


def label_to_channel(labels, num_classes=19):
    # map value to each category
    #
    # input:
    #   label (n, h, w)
    #
    # output:
    #   (n, num_classes, h, w)
    n, h, w = labels.shape
    label_channel = torch.zeros(n, num_classes, h, w)
    for i in range(num_classes):
        label_channel[:, i] = (labels == i) * 1
    return label_channel.cuda(args.gpu)


def main():
    """Create the model and start the training."""

    LOSS_OPTION = 'bce'

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)
    print("input size =", input_size)
    print("input_size_target =", input_size_target)
    cudnn.enabled = True
    gpu = args.gpu

    # tensorboard logger
    configure("check_output/log")

    # Create network
    if args.model == 'DeepLab':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # print(i)
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print(i_parts)
        # new_params = saved_state_dict
        model.load_state_dict(new_params)

    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # init D
    # model_D1 = XiaoDiscriminator(num_classes=args.num_classes)
    # model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D1 = XiaoAttentionDiscriminator(num_classes=args.num_classes)
    model_D1.train()
    model_D1.cuda(args.gpu)

    # ===============  Preprocess and Load Data ==============

    # create dir
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    pdb.set_trace()
    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    pdb.set_trace()

    trainloader_iter = enumerate(trainloader)
    pdb.set_trace()
    aa, bb = next(trainloader_iter)
    a, b, c, d = bb

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)

    pdb.set_trace()

    targetloader_iter = enumerate(targetloader)
    pdb.set_trace()
    # implement model.optim_parameters(args) to handle different models' lr setting

    # LEARNING_RATE_G = 2.5e-4
    # LEARNING_RATE_D = 1e-4
    # args.learning_rate_G = LEARNING_RATE_G
    # args.learning_rate_D = LEARNING_RATE_D
    # optimizer = optim.SGD(model.optim_parameters(args),
    #                      lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.optim_parameters(args),
    #                       lr=LEARNING_RATE_G, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1 = optim.Adam(filter(lambda p: p.requires_grad, model_D1.parameters()), lr=args.learning_rate_D, betas=(0.9, 0.99))
    # next try

    # optimizer_D1 = optim.Adam(filter(lambda p: p.requires_grad, model_D1.parameters()), lr=args.learning_rate_D,
    #                           betas=(0.0, 0.9))
    # optimizer_D1 = optim.Adam(filter(lambda p: p.requires_grad, model_D1.parameters()), lr=LEARNING_RATE_D,
    #                           betas=(0.9, 0.99))
    # TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
    # optimizer_D1 = optim.Adam(filter(lambda p: p.requires_grad, model_D1.parameters()), lr=args.lr, betas=(0, 0.99))

    optimizer_D1.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    # INPUT_SIZE = '1280,720'
    interp = nn.Upsample(size=(input_size[1], input_size[0]), align_corners=False, mode='bilinear')
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), align_corners=False, mode='bilinear')
    # inter_mini = nn.Upsample(size=(int(input_size[1]/2), int(input_size[0]/2)), align_corners=False, mode='bilinear')
    # inter_mini = nn.Upsample(size=(int(input_size[1]), int(input_size[0])), align_corners=False, mode='bilinear')
    inter_mini = nn.Upsample(size=(input_size[1]/8, input_size[0]/8), align_corners=False, mode='bilinear')



    # labels for adversarial training
    source_label, target_label = 0, 1

    for i_iter in range(args.num_steps):

        loss_D_value = 0
        loss_source = 0
        loss_target = 0

        #  opt G
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        #  opt D
        optimizer_D1.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)


        for sub_i in range(args.iter_size):



            # ============================================== #
            #                   Train G                      #
            # ============================================== #

            # Enable G backpropgation requires_grad
            # for param in model.parameters():
            #     param.requires_grad = True
            # Disable D backpropgation requires_grad
            for param in model_D1.parameters():
                param.requires_grad = False

            # ===================================== #
            #         Train G source                #
            # ===================================== #

            _, batch = trainloader_iter.__next__()
            images, labels, _, names = batch
            images = Variable(images).cuda(args.gpu)
            label = Variable(labels).cuda(args.gpu)
            pred_source_real = model(images)
            # resize to source size
            pred_source_real = interp(pred_source_real)

            # seg loss
            loss_seg1 = loss_calc(pred_source_real, label, args.gpu)
            loss = loss_seg1
            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_source += loss_seg1.data.cpu().numpy() / args.iter_size

            # ===================================== #
            #         Train G target                #
            # ===================================== #


            _, batch = targetloader_iter.__next__()
            images, _, _, target_name = batch
            images = Variable(images).cuda(args.gpu)
            pred_target_fake = model(images)

            pred_target_fake = interp_target(pred_target_fake)


            # d_out_fake = model_D1(F.softmax(pred_target1), label=mini_target_image)
            d_out_fake = model_D1(F.softmax(pred_target_fake), inter_mini(F.softmax(pred_target_fake)))

            # d_loss_fake = bce_loss(d_out_fake,
            #                        Variable(torch.FloatTensor(d_out_fake.data.size()).fill_(source_label)).cuda(
            #                            args.gpu))

            # use hinge loss
            if LOSS_OPTION == 'wgan-gp':
                d_loss_fake = - d_out_fake.mean()
            elif LOSS_OPTION == 'hinge':
                d_loss_fake = - d_out_fake.mean()
            elif LOSS_OPTION == 'bce':
                d_loss_fake = bce_loss(d_out_fake,
                                       Variable(torch.FloatTensor(d_out_fake.data.size()).fill_(source_label)).cuda(
                                           args.gpu))
            LAMBDA_ADV_G = 0.001

            loss = LAMBDA_ADV_G * d_loss_fake / args.iter_size
            loss.backward()
            loss_target += d_loss_fake.data.cpu().numpy() / args.iter_size

            # ============================================== #
            #                   Train d                      #
            # ============================================== #

            # Disable G backpropgation requires_grad
            # for param in model.parameters():
            #     param.requires_grad = False
            # Enable D backpropgation requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            # ===================================== #
            #         Train D source                #
            # ===================================== #

            # _, batch = trainloader_iter.__next__()
            # images, labels, _, names = batch
            # images = Variable(images).cuda(args.gpu)
            # label = Variable(labels).cuda(args.gpu)
            #
            # pred_source_real = model(images).detach()
            pred_source_real = pred_source_real.detach()
            # resize to source size
            # pred_source_real = interp(pred_source_real)
            # pred_source_real = interp(pred_source_real)
            # resize to mini size for inner product
            # mini_source_image = inter_mini(images)
            # d_out_real = model_D1(F.softmax(pred_source_real), label=mini_source_image)
            d_out_real = model_D1(F.softmax(pred_source_real), label=inter_mini(label_to_channel(label)))

            if LOSS_OPTION == 'wgan-gp':
                d_loss_real = - d_out_real.mean()
            elif LOSS_OPTION == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            elif LOSS_OPTION == 'bce':
                d_loss_real = bce_loss(d_out_real,
                                       Variable(torch.FloatTensor(d_out_real.data.size()).fill_(source_label)).cuda(
                                           args.gpu))

            # d_out_real = model_D1(F.softmax(pred_source_real))
            # d_loss_real = bce_loss(d_out_real,
            #                    Variable(torch.FloatTensor(d_out_real.data.size()).fill_(source_label)).cuda(args.gpu))

            loss = d_loss_real / args.iter_size / 2
            loss.backward()
            loss_D_value += d_loss_real.data.cpu().numpy() / args.iter_size

            # ===================================== #
            #         Train D target                #
            # ===================================== #
            # _, batch = targetloader_iter.__next__()
            # images, _, _, target_name = batch
            # images = Variable(images).cuda(args.gpu)
            # pred_target_fake = model(images).detach()
            # resize to source size
            # pred_target_fake = interp(pred_target_fake)
            # pred_target_fake = interp(pred_target_fake)

            # resize to mini size for inner product
            # mini_target_image = inter_mini(images)

            # d_out_fake = model_D1(F.softmax(pred_target_fake), label=mini_target_image)
            pred_target_fake = pred_target_fake.detach()
            d_out_fake = model_D1(F.softmax(pred_target_fake), label=inter_mini(F.softmax(pred_target_fake)))
            # d_out_fake = model_D1(F.softmax(pred_target_fake))
            #
            #

            if LOSS_OPTION == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif LOSS_OPTION == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            elif LOSS_OPTION == 'bce':
                d_loss_fake = bce_loss(d_out_fake,
                                       Variable(torch.FloatTensor(d_out_fake.data.size()).fill_(target_label)).cuda(
                                           args.gpu))
            loss = d_loss_fake / args.iter_size / 2
            loss.backward()
            loss_D_value += d_loss_fake.data.cpu().numpy() / args.iter_size / 2

            if i_iter % 50 == 0 and sub_i == args.iter_size - 1:
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

        optimizer.step()
        optimizer_D1.step()
        # print('exp = {}'.format(args.snapshot_dir))

        print("iter = {0:8d}/{1:8d}, loss_G_source_1 = {2:.3f} loss_G_adv1 = {3:.3f} loss_D1 = {4:.3f}".format(
                i_iter, args.num_steps, loss_source, float(loss_target), float(loss_D_value)))




        # save checkpoint
        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '_D1.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking pth in shapshot dir ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))


if __name__ == '__main__':
    main()
