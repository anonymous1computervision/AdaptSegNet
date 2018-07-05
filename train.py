"""
Code concept modify from AdaptSegNet
    - Learning to Adapt Structured Output Space for Semantic Segmentation
    - https://arxiv.org/abs/1802.10349
    - https://github.com/wasidennis/AdaptSegNet

Code style modify Modify from MUNIT
    - https://github.com/NVlabs/MUNIT

Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import pdb

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils import data

from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from trainer import AdaptSeg_Trainer
from attn_trainer import AdaptSeg_Attn_Trainer
from mini_trainer import Mini_AdaptSeg_Trainer
from util import get_all_data_loaders, get_config

def main():

    # cuda setting
    cudnn.enabled = True
    cudnn.benchmark = True
    # config setting
    # CONFIG_PATH = "./configs/attention_v7_attn.yaml"
    CONFIG_PATH = "./configs/default-mini.yaml"
    # CONFIG_PATH = "./configs/attention_v1.yaml"

    config = get_config(CONFIG_PATH)

    gpu = config["gpu"]
    # training setting
    num_steps = config['num_steps']

    # log setting
    test_summary = config["test_summary"]
    snapshot_save_iter = config["snapshot_save_iter"]
    image_save_iter = config["image_save_iter"]

    log_dir = config["log_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    snapshot_save_dir = config["snapshot_save_dir"]
    if not os.path.exists(snapshot_save_dir):
        os.makedirs(snapshot_save_dir)

    image_save_dir = config["image_save_dir"]
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    # data loader
    train_loader, target_loader = get_all_data_loaders(config)


    # model init
    if config["g_model"] == "attn":
        trainer = AdaptSeg_Attn_Trainer(config)
    if config["g_model"] == "mini":
        trainer = Mini_AdaptSeg_Trainer(config)
    else:
        trainer = AdaptSeg_Trainer(config)
    # trainer.cuda(gpu)

    if config["restore"]:
        trainer.restore(model_name=config["model"], num_classes=config["num_classes"], restore_from=config["restore_from"])




    # Start training
    while True:
        for i_iter, (train_batch, target_batch) in enumerate(zip(train_loader, target_loader)):
            # if memory issue can clear cache
            # torch.cuda.empty_cache()

            trainer.init_each_epoch(i_iter)
            trainer.update_learning_rate()

            # ====================== #
            #   Main training code   #
            # ====================== #

            # train G use source image
            src_images, labels, _, names = train_batch
            # print("get source image shape", src_images.shape)
            # print("get source labels shape", labels.shape)
            src_images = Variable(src_images).cuda(gpu)
            trainer.gen_source_update(src_images, labels, names)
            del src_images

            # train G use target image
            target_images, _, _, target_name = target_batch
            target_images = Variable(target_images).cuda(gpu)
            trainer.gen_target_update(target_images, target_name)
            del target_images

            # # train discriminator use prior generator image
            trainer.dis_update(labels=labels)

            # show log
            trainer.show_each_loss()

            # save image to check
            if i_iter % image_save_iter == 0:
                print("image_save_dir", image_save_dir)
                trainer.snapshot_image_save(dir_name=image_save_dir)

            # save checkpoint .pth
            if i_iter % snapshot_save_iter == 0 and i_iter > 0:
                # print("save model")
                trainer.save_model(snapshot_save_dir=snapshot_save_dir)

            # save final model .pth
            if i_iter == num_steps - 1:
                trainer.save_model(snapshot_save_dir=snapshot_save_dir)

if __name__ == "__main__":
    main()