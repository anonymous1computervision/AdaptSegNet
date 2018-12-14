import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image


class GTA5DataSet(data.Dataset):
    # def __init__(self, root, list_path, max_iters=None, crop_size=(720, 1280), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, is_train=True):
    def __init__(self, root, list_path, max_iters=None, crop_size=600, mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, is_train=True):

        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.is_train = is_train
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        random.shuffle(self.files)
        num_sample = len(self.files)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))
        print("# Crop Size: {}".format(self.crop_size))

    def __len__(self):
        return len(self.files)

    def _scale(self, img, seg):
        h_s, w_s = 720, 1312
        img_scale = img.resize((w_s, h_s), Image.BICUBIC)
        seg_scale = seg.resize((w_s, h_s), Image.NEAREST)

        return img_scale, seg_scale

    def _crop(self, img, seg, cropSize, is_train):
        h_s, w_s = 720, 1312
        if is_train:
            # random crop
            x1 = random.randint(0, w_s - cropSize)
            y1 = random.randint(0, h_s - cropSize)
            img_crop = img[y1: y1 + cropSize, x1: x1 + cropSize, :]
            seg_crop = seg[y1: y1 + cropSize, x1: x1 + cropSize]

        else:
            # no crop
            img_crop = img
            seg_crop = seg
        return img_crop, seg_crop

    def _flip(self, img, seg):
        img_flip = img[:, ::-1, :]
        seg_flip = seg[:, ::-1]
        return img_flip, seg_flip

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        # image = image.resize(self.crop_size, Image.BICUBIC)
        # label = label.resize(self.crop_size, Image.NEAREST)


        # 1214 add for data augmentation
        # copy from https://github.com/kashyap7x/Semantic-Input-Masking/blob/master/dataset.py
        image, label = self._scale(image, label)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        # random crop
        image, label = self._crop(image, label, self.crop_size, self.is_train)

        # random flip
        if self.is_train and random.choice([-1, 1]) > 0:
            image, label = self._flip(image, label)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
