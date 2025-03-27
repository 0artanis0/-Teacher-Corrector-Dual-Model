# coding:utf8
"""
定义了数据集类，CarDateSet和PredictionDateSet分别用于训练和预测

    @Project: Dataset_label_error_cleanup_classifier
    @File   : dataset.py
    @Author : Hongli Zhao
    @E-mail : zhaohongli8711@outlook.com
    @Date   : 2023-12-22 10:04:28
"""

import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torch


class PredictionDateSet(data.Dataset):
    """用于预测的数据集类。

    Args:
        root (str): 数据集根目录。
        lists (str): 包含图像路径和标签的列表文件路径。
        transforms (torchvision.transforms.Compose, optional): 数据预处理的转换操作，默认为None。
        train (bool): 指示是否为训练集，默认为True。
        test (bool): 指示是否为测试集，默认为False。
    """

    def __init__(self, root, lists, transforms=None, train=True, test=False):
        self.test = test

        with open(lists, 'r') as f:
            lines = f.readlines()

        imgs = []
        labels = []

        for line in lines:
            b = line.split()[0]
            imgs.append(os.path.join(root, b))
            a = int(line.split()[1])
            labels.append(a)

        self.imgs = imgs
        self.labels = labels

        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        label = self.labels[index]
        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        return data, img_path

    def __len__(self):
        return len(self.imgs)


class CarDateSet(data.Dataset):
    """用于训练的数据集类。

    Args:
        root (str): 数据集根目录。
        lists (str): 包含图像路径和标签的列表文件路径。
        transforms (torchvision.transforms.Compose, optional): 数据预处理的转换操作，默认为None。
        train (bool): 指示是否为训练集，默认为True。
        test (bool): 指示是否为测试集，默认为False。
    """

    def __init__(self, root, lists, transforms=None, train=True, test=False):
        self.test = test

        with open(lists, 'r') as f:
            lines = f.readlines()

        imgs = []
        labels = []

        for line in lines:
            b = line.split()[0]
            imgs.append(os.path.join(root, b))
            a = int(line.split()[1])
            labels.append(a)

        self.imgs = imgs
        self.labels = labels

        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((227, 227)),
                T.RandomCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        label = self.labels[index]
        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = CarDateSet('./data/train', './data/train.txt')
    img, label = dataset[0]
    for img, label in dataset:
        print(img.size(), img.float().mean(), label)
