import sys
import os

import torch
from torch.autograd import Variable
from torch.utils import data, model_zoo
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn

# from model.deeplab_single import Res_Deeplab as Res_Deeplab_S
from model.deeplab_multi import Res_Deeplab as Res_Deeplab_M

from dataset.my_test_dataloader import MyDataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './data/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
DATA_TEST_DIRECTORY = './my_test_img'
DATA_TEST_LIST_PATH = './my_test_img/val.txt'

IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 100000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
TARGET = 'cityscapes'
SET = 'val'
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    # parser.add_argument("--save", type=str, default=SAVE_PATH,
    #                     help="Path to save result.")
    return parser.parse_args()
if __name__ == '__main__':

    """Create the model and start the evaluation process."""

    args = get_arguments()

    gpu0 = args.gpu

    model = Res_Deeplab(num_classes=args.num_classes)

    if args.restore_from[:4] == 'http':
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(
        MyDataSet(DATA_TEST_DIRECTORY, DATA_TEST_LIST_PATH, crop_size=(1024, 512), mean=IMG_MEAN, scale=False,
                          mirror=False, set=args.set),
        batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear')
    save_path = os.path.join(DATA_TEST_DIRECTORY, "output")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd' % index)
        image, _, _, name = batch
        # output1 = model(Variable(image, volatile=True).cuda(gpu0)).detach()
        output1, _ = model(Variable(image, volatile=True).cuda(gpu0))

        output = interp(output1).cpu().data[0].numpy()

        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]

        # output.save('%s/%s' % (save_path, name))
        output_col.save('%s/%s_color.png' % (save_path, name.split('.')[0]))

