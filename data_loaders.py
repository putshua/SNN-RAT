import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST, SVHN
import warnings
import os
from torch.utils.data import random_split
from os import listdir
import numpy as np
from os.path import isfile, join
import json


def build_cifar(cutout=True, use_cifar10=True, download=True):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()]

    if use_cifar10:
        # aug.append(
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = CIFAR10(root='E:\datasets',
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root='E:\datasets',
                              train=False, download=download, transform=transform_test)
        norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        # aug.append(
        #     transforms.Normalize(
        #         (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        # )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='E:\datasets',
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root='E:\datasets',
                               train=False, download=download, transform=transform_test)
        norm = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    return train_dataset, val_dataset, norm