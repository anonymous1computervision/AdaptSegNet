import argparse
from os.path import join

import numpy as np
import json

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
import torch.backends.cudnn as cudnn


# from model.deeplab_multi import Res_Deeplab
# from model.deeplab_single import Res_Deeplab
# from model.deeplab_single_attention import Res_Deeplab
# from model.networks import StyleEncoder, MLP
# from model.deeplab_multi import Res_Deeplab
# from model.deeplab_single_IN import Res_Deeplab
from in_trainer import AdaptSeg_IN_Trainer
from util import get_all_data_loaders, get_config

from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from os.path import join
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Cityscapes/data'
DATA_LIST_PATH = './my_test_img/val_mini.txt'
SAVE_PATH = './my_test_img/val_mini'

IGNORE_LABEL = 255
NUM_CLASSES = 19
SET = 'val'

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

def get_test_mini_set():
    """Create the model and start the evaluation process."""
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    testloader = data.DataLoader(
        cityscapesDataSet(DATA_DIRECTORY, DATA_LIST_PATH, crop_size=(1024, 512), mean=IMG_MEAN, scale=False,
                          mirror=False, set=SET),
        batch_size=1, shuffle=False, pin_memory=True)
    return testloader

def output_to_image(image, name):
    interp = nn.Upsample(size=(1024, 2048), mode='bilinear')
    output = interp(image).cpu().data[0].numpy()

    output = output.transpose(1,2,0)
    output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

    output_col = colorize_mask(output)
    output = Image.fromarray(output)

    name = name[0].split('/')[-1]
    output.save('%s/%s' % (SAVE_PATH, name))
    output_col.save('%s/%s_color.png' % (SAVE_PATH, name.split('.')[0]))

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir="./data/Cityscapes/data/gtFine/val", pred_dir="my_test_img/val_mini", devkit_dir='./my_test_img'):
    """
        Compute IoU given the predicted colorized images and
        """

    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val_mini.txt')
    label_path_list = join(devkit_dir, 'label_mini.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()), gt_imgs[ind],
                                                                                  pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # if ind > 0 and ind % 10 == 0:
        #     print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)
    record_string = ""
    for ind_class in range(num_classes):
        record_string += '===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)) + '\n'
    total_miou = round(np.nanmean(mIoUs) * 100, 2)
    record_string += '===> mIoU: ' + str(total_miou) + '\n'
    # print(record_string)


    return total_miou, record_string