import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10


def cifar10():
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = CIFAR10(root='E:\datasets',
                                train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root='E:\datasets',
                              train=False, download=True, transform=transform_test)
        norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        return train_dataset, val_dataset, norm