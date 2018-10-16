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
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils import data
from PIL import Image

from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from trainer import AdaptSeg_Trainer
from attn_trainer import AdaptSeg_Attn_Trainer
from mini_trainer import Mini_AdaptSeg_Trainer
from in_trainer import AdaptSeg_IN_Trainer
from trainer_edge_aux import AdaptSeg_Edge_Aux_Trainer
from trainer_edge_aux_sn import AdaptSeg_Edge_Aux_SN_Trainer
from trainer_edge_aux_deeplab_v3 import AdaptSeg_Edge_Aux_v3_Trainer
from trainer_PSP_edge_aux import AdaptSeg_PSP_Edge_Aux_Trainer

from trainer_scratch_gen import DeepLab_Scratch_Trainer
from dense_trainer import DenseSeg_Trainer
from util import get_all_data_loaders, get_config
from test_iou import output_to_image, compute_mIoU, get_test_mini_set

def main():

    # cuda setting
    cudnn.enabled = True
    cudnn.benchmark = True
    # config setting
    # CONFIG_PATH = "./configs/attention_v7_attn.yaml"
    # CONFIG_PATH = "./configs/default-mini.yaml"
    # CONFIG_PATH = "./configs/default-in-bce-v3.yaml"
    # CONFIG_PATH = "./configs/default_v2.yaml"
    # CONFIG_PATH = "./configs/default_deeplab_v3_plus.yaml"
    # CONFIG_PATH = "./configs/default-hinge-v7.yaml"
    # CONFIG_PATH = "./configs/default-in-hinge-v5.yaml"
    # CONFIG_PATH = "./configs/default-in.yaml"
    # CONFIG_PATH = "./configs/default_edge.yaml"
    # CONFIG_PATH = "./configs/default_edge_deeplabv3.yaml"
    # CONFIG_PATH = "./configs/default_edge_TTUR.yaml"
    CONFIG_PATH = "./configs/default_PSPNet_edge_TTUR.yaml"
    # CONFIG_PATH = "./configs/default_edge_TTUR_D_beta.yaml"
    # CONFIG_PATH = "./configs/default__SA_TTUR_D_fore_beta.yaml"

    # CONFIG_PATH = "./configs/default_edge_SN_TTUR.yaml"

    # CONFIG_PATH = "./configs/default-fc-dense.yaml"
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
    elif config["g_model"] == "mini":
        trainer = Mini_AdaptSeg_Trainer(config)
    elif config["g_model"] == "in":
        trainer = AdaptSeg_IN_Trainer(config)
        print("use AdaptSeg_IN_Trainer")
    elif config["model"] == "FC-DenseNet":
        trainer = AdaptSeg_Trainer(config)
        print("use FC-DenseNet")
    elif config["model"] == "DeepLab_v3_plus":
        trainer = AdaptSeg_Trainer(config)
        # trainer = DeepLab_Scratch_Trainer(config)
        print("use DeepLab_v3_plus")
    elif config["model"] == "DeepLabEdge":
        trainer = AdaptSeg_Edge_Aux_Trainer(config)
        # trainer = DeepLab_Scratch_Trainer(config)
        print("use DeepLabEdge")
    elif config["model"] == "DeepLabEdgeSN":
        trainer = AdaptSeg_Edge_Aux_SN_Trainer(config)
        # trainer = DeepLab_Scratch_Trainer(config)
        print("use DeepLabEdgeSN")
        print("use DeepLabEdgeSN")
        print("use DeepLabEdgeSN")
    elif config["model"] == "DeepLabv3+":
        trainer = AdaptSeg_Edge_Aux_v3_Trainer(config)
        # trainer = DeepLab_Scratch_Trainer(config)
        print("use DeepLabv3")
        print("use DeepLabv3")
        print("use DeepLabv3")
    elif config["model"] == "PSPNet":
        print("use PSPNet")
        print("use PSPNet")
        print("use PSPNet")
        trainer = AdaptSeg_PSP_Edge_Aux_Trainer(config)
    else:
        trainer = AdaptSeg_Trainer(config)

    # todo: remove this line without dev version
    assert config["model"] == "PSPNet"

    # trainer.cuda(gpu)
    print("config[restore] =", config["restore"])
    print("config[model]  =", config["model"])
    if config["restore"] and config["restore_from"] != "None":
        trainer.restore(model_name=config["model"], num_classes=config["num_classes"], restore_from=config["restore_from"])
    # if config["restore"] and config["model"] == "DeepLab_v3_plus":
    #     print(" in restore deeplab v3")
    #     trainer.restore(model_name=config["model"], num_classes=config["num_classes"], restore_from=config["restore_from"])

    # restore_from = ".\snapshots\GTA2Cityscapes_multi\GTA5_14000_trainer_all.pth"
    # print(str(trainer.state_dict())[:100])
    # saved_state_dict = torch.load(restore_from)
    # trainer.load_state_dict(saved_state_dict)
    # print("after model load")

    # todo:next time need to remove
    # trainer.restore_D()
    record_each_epoch = ""
    best_score_record = {
        "epochs": 0,
        "total_mIOU": 0,
        "recording_string": ""
    }
    # train_only_src = True
    # Start training
    while True:
        for i_iter, (train_batch, target_batch) in enumerate(zip(train_loader, target_loader)):
            # if memory issue can clear cache
            torch.cuda.empty_cache()
            trainer.init_each_epoch(i_iter)
            # todo:scratch train so temporally out comment
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
            # torch.cuda.empty_cache()

            # train G use target image
            target_images, _, _, target_name = target_batch
            target_images = Variable(target_images).cuda(gpu)
            #trainer.gen_target_and_foreground_update(target_images, target_name)

            trainer.gen_target_update(target_images, target_name)
            # trainer.gen_target_foreground_update(target_images, target_name)

            del target_images
            # torch.cuda.empty_cache()

            # train discriminator use prior generator image
            trainer.dis_update(labels=labels)
            # trainer.dis_foreground_update(labels=labels)


            # train G use weakly label
            # trainer.gen_weakly_update(target_images, target_name)


            # show log
            trainer.show_each_loss()


            # save image to check
            if i_iter % image_save_iter == 0:
                print("image_save_dir", image_save_dir)
                trainer.snapshot_image_save(dir_name=image_save_dir)
                trainer.snapshot_edge_save(dir_name=image_save_dir, labels=labels)

            # save checkpoint .pth
            if i_iter % snapshot_save_iter == 0:
            # if i_iter % snapshot_save_iter == 0 and i_iter > 0:
            # if i_iter % snapshot_save_iter == 0:
                # print("save model")
                trainer.save_model(snapshot_save_dir=snapshot_save_dir)
                torch.save(trainer.state_dict(),
                           os.path.join(snapshot_save_dir, 'GTA5_' + str(i_iter) + '_trainer_all.pth'))

            # save final model .pth
            if i_iter == num_steps - 1:
                trainer.save_model(snapshot_save_dir=snapshot_save_dir)


            # test image to check and get mIOU
            # if i_iter % 100 == 0 and i_iter > 0:
            if i_iter % 100 == 0:
                # torch.cuda.empty_cache()
                testloader = get_test_mini_set()
                # trainer.eval()
                with torch.no_grad():
                    for index, batch in enumerate(testloader):
                        # if memory issue can clear cache
                        image, _, _, name = batch
                        output = trainer(Variable(image).cuda())
                        # maybe 2 output (out, auxiluary)
                        # but i modify trainer forward
                        # if isinstance(output, tuple):
                        #     output = output[0]
                        output_to_image(output, name)
                    total_miou, recording_string = compute_mIoU()

                    # recording_total = f"Test summary = %s\n\n" \
                    #                   "========= Best score =========\n" \
                    #                   "best epoches = %s\n" \
                    #                   "%s\n\n" %(
                    #                     config["test_summary"],
                    #                     best_score_record["epochs"],
                    #                     best_score_record["recording_string"])
                    recording_total = f"Test summary = %s\n\n" \
                                      "========= Best score =========\n" \
                                      "best epoches = %s\n" \
                                      "gamma = %.5f\n"\
                                      "%s\n\n" %(
                                        config["test_summary"],
                                        best_score_record["epochs"],
                                        trainer.discriminator_gamma,
                                        best_score_record["recording_string"])
                    # currenet_result = f"\n========= Current score =========\n" \
                    #                   "epoches = % s\n" \
                    #                   "%s" \
                    #                   % (i_iter,
                    #                      recording_string)
                    currenet_result = f"\n========= Current score =========\n" \
                                      "epoches = % s\n" \
                                      "gamma = %.5f\n" \
                                      "%s" \
                                      % (i_iter,
                                         trainer.discriminator_gamma,
                                         recording_string)
                    recording_total += currenet_result
                    print(recording_total)

                    if i_iter % 1000 == 0 or i_iter <= 1000:
                        record_each_epoch += f"\n========= epoch score  = % s =========\n" \
                                         "gamma = %.5f\n" \
                                         "%s\n" \
                                          % (i_iter,
                                             trainer.discriminator_gamma,
                                             recording_string)
                    recording_total += record_each_epoch
                            # recording_total = f"Test summary = %s\n\n"\
                            #         "========= Best score =========\n"\
                            #         "best epoches = %s\n"\
                            #         "%s\n\n"\
                            #         "========= Current score =========\n" \
                            #         "epoches = % s\n"\
                            #         "%s"\
                            #          % (config["test_summary"],
                            #             best_score_record["epochs"],
                            #             best_score_record["recording_string"],
                            #             i_iter,
                            #             recording_string)


                    with open('./result.txt', 'w') as f:
                        f.write(recording_total)


                    # if higher accuracy in evaluate
                    if(total_miou > best_score_record["total_mIOU"]):
                        print("this epoch get higher accuracy")
                        best_score_record["epochs"] = i_iter
                        best_score_record["total_mIOU"] = total_miou
                        best_score_record["recording_string"] = recording_string


                    # trainer.train()
                    # del testloader
                    # torch.cuda.empty_cache()


if __name__ == "__main__":
    main()