from collections import OrderedDict
from typing import Union, List

import torch
import torchvision
import torchvision.transforms as transforms

from constants import DATA_ROOT_PATH


class DatasetName:
    CIFAR10 = 'CIFAR10'
    MNIST = 'MNIST'
    FashionMNIST = 'FASHIONMNIST'


def get_dataset(dataset: str = "CIFAR10", validation_set_pc=0.05):
    if dataset.upper() == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT_PATH, train=False, download=True, transform=transform)
        train_val_set = torchvision.datasets.CIFAR10(root=DATA_ROOT_PATH, train=True, download=True,
                                                     transform=transform)

        train_set, val_set = make_train_val_sets(train_val_set, validation_set_pc)
    elif dataset.upper() == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        test_set = torchvision.datasets.MNIST(root=DATA_ROOT_PATH, train=False,
                                              transform=transform, download=True)
        train_val_set = torchvision.datasets.MNIST(root=DATA_ROOT_PATH, train=True,
                                                   transform=transform, download=True)

        train_set, val_set = make_train_val_sets(train_val_set, validation_set_pc)
    elif dataset.upper() == "FASHIONMNIST":
        test_set = torchvision.datasets.FashionMNIST(root=DATA_ROOT_PATH, train=False,
                                                     download=True, transform=transforms.ToTensor())
        train_val_set = torchvision.datasets.FashionMNIST(root=DATA_ROOT_PATH, train=True,
                                                          download=True, transform=transforms.ToTensor())

        train_set, val_set = make_train_val_sets(train_val_set, validation_set_pc)
    else:
        raise Exception(f"Dataset - {dataset} -  is not available, you can add it at data.py!")

    return train_set, val_set, test_set


def get_dataset_info(dataset="CIFAR10", info_keys: Union[str, List[str]] = None):
    """
    Returns requested information in order if info_keys specified, else returns
    all dataset information in the order: image_dim, n_outputs, classes.
    """
    dataset_info = {
        "CIFAR10": OrderedDict([
            ("image_dim", (3, 32, 32)),
            ("n_outputs", 10),
            ("classes", ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'))
        ]),
        "MNIST": OrderedDict([
            ("image_dim", (1, 28, 28)),
            ("n_outputs", 10),
            ("classes", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
        ]),
        "FASHIONMNIST": OrderedDict([
            ("image_dim", (1, 28, 28)),
            ("n_outputs", 10),
            ("classes", ('T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                         'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'))
        ])
    }

    dataset = dataset.upper()
    if info_keys is not None:
        if isinstance(info_keys, list):
            info_keys = [info_key.upper() for info_key in info_keys]
            return [dataset_info[dataset][key] for key in info_keys]
        if isinstance(info_keys, str):
            key = info_keys
            return dataset_info[dataset][key]

    return list(dataset_info[dataset].values())


def make_train_val_sets(train_val_set, validation_set_pc):
    train_size = int((1 - validation_set_pc) * len(train_val_set))
    val_size = len(train_val_set) - train_size
    train_set, val_set = torch.utils.data.random_split(train_val_set, [train_size, val_size])
    return train_set, val_set


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    image_dim, n_outputs, classes = get_dataset_info("mnist")
    print(image_dim)
