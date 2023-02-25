import os.path
import pathlib
import tarfile
import urllib.request
from pathlib import Path

import numpy
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from constants import DATA_ROOT_PATH
from utils import show_image


class DatasetName:
    CIFAR10 = 'CIFAR10'
    CIFAR100 = "CIFAR100"
    MNIST = 'MNIST'
    FashionMNIST = 'FASHIONMNIST'
    OxfordFlowers = "OXFORDFLOWERS-102"
    Oxford3Pet = "OXFORD3TPET"


class CustomResize(transforms.Resize):
    def __init__(self, size, dont_resize=False):
        super().__init__(size)
        self.dont_resize = dont_resize

    def forward(self, img):
        if self.dont_resize:
            return img

        resize = transforms.Resize(self.size, self.interpolation, self.max_size, self.antialias)
        return resize(img)


class DataMaker:
    def __init__(self, dataset, model_ip_dims, validation_set_pc=0.05):
        self.dataset = dataset
        self.model_ip_dims = model_ip_dims
        self.validation_set_pc = validation_set_pc

        self.train_set, self.val_set, self.test_set = self.make_dataset_sets(
            model_ip_dims[1:], dont_resize=False)

        print(self.get_img_dims())

        # DIFFERENT LOGIC
        # self.train_set, self.val_set, self.test_set = self.make_dataset_sets(
        #     model_ip_dims[1:], dont_resize=True)

        # Resize if image dims are smaller than min model dims.
        # if self.get_img_dims()[1] < model_ip_dims[1]:
        #     print("Resizing . . .")
        #     self.train_set, self.val_set, self.test_set = self.make_dataset_sets(
        #         model_ip_dims[1:], dont_resize=False)


    def make_loaders(self, batch_size, num_workers=1):
        train_loader = DataLoader(self.train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(self.val_set, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(self.test_set, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader

    def get_classes(self):
        if self.dataset == DatasetName.OxfordFlowers:
            return self.train_set.classes
        return self.train_set.dataset.classes

    def get_num_outputs(self):
        if self.dataset == DatasetName.OxfordFlowers:
            return len(self.train_set.classes)
        return len(self.train_set.dataset.classes)

    def get_img_dims(self):
        return list(self.train_set[0][0].shape)

    def get_dataset_sets(self):
        return self.train_set, self.val_set, self.test_set

    def make_dataset_sets(self, model_ip_dims, dont_resize):
        if self.dataset.upper() == DatasetName.CIFAR10:
            train_set, val_set, test_set = self._get_cifar10_sets(
                model_ip_dims, dont_resize)

        elif self.dataset == DatasetName.CIFAR100:
            train_set, val_set, test_set = self._get_cifar100_sets(
                model_ip_dims, dont_resize)

        elif self.dataset == DatasetName.MNIST:
            train_set, val_set, test_set = self._get_mnist_sets(
                model_ip_dims, dont_resize)

        elif self.dataset == DatasetName.FashionMNIST:
            train_set, val_set, test_set = self._get_fashion_mnist_sets(
                model_ip_dims, dont_resize)

        elif self.dataset == DatasetName.OxfordFlowers:
            train_set, val_set, test_set = self._get_oxford_flowers_sets(
                model_ip_dims, dont_resize)

        elif self.dataset == DatasetName.Oxford3Pet:
            train_set, val_set, test_set = self._get_oxford_pet_sets(
                model_ip_dims, dont_resize)
        else:
            raise Exception(f"Dataset {self.dataset} not available!")

        return train_set, val_set, test_set

    def _get_cifar10_sets(self, new_dims, dont_resize):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
            CustomResize(new_dims, dont_resize)
        ])

        test_set = torchvision.datasets.CIFAR10(
            root=DATA_ROOT_PATH, train=False, download=True,
            transform=transform)
        train_val_set = torchvision.datasets.CIFAR10(
            root=DATA_ROOT_PATH, train=True, download=True,
            transform=transform)

        train_set, val_set = make_train_val_sets(
            train_val_set, self.validation_set_pc)

        return train_set, val_set, test_set

    def _get_cifar100_sets(self, new_dims, dont_resize):

        mean, std = ((0.5029764, 0.49406436, 0.49055508),
                     (0.29286674, 0.28687882, 0.28227282))
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            CustomResize(new_dims, dont_resize)
        ])
        mean, std = ((0.4890062, 0.47970363, 0.47680542),
                     (0.264582, 0.258996, 0.25643882))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            CustomResize(new_dims, dont_resize)
        ])

        test_set = torchvision.datasets.CIFAR100(
            root=DATA_ROOT_PATH, train=False, download=True,
            transform=transform_test)
        train_val_set = torchvision.datasets.CIFAR100(
            root=DATA_ROOT_PATH, train=True, download=True,
            transform=transform_train)

        train_set, val_set = make_train_val_sets(
            train_val_set, self.validation_set_pc)

        return train_set, val_set, test_set

    def _get_mnist_sets(self, new_dims, dont_resize):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            CustomResize(new_dims, dont_resize)
        ])

        test_set = torchvision.datasets.MNIST(
            root=DATA_ROOT_PATH, train=False,
            transform=transform, download=True)
        train_val_set = torchvision.datasets.MNIST(
            root=DATA_ROOT_PATH, train=True,
            transform=transform, download=True)

        train_set, val_set = make_train_val_sets(
            train_val_set, self.validation_set_pc)

        return train_set, val_set, test_set

    def _get_fashion_mnist_sets(self, new_dims, dont_resize):
        transform = transforms.Compose([
            transforms.ToTensor(),
            CustomResize(new_dims, dont_resize)
        ])

        test_set = torchvision.datasets.FashionMNIST(
            root=DATA_ROOT_PATH, train=False,
            download=True, transform=transform)
        train_val_set = torchvision.datasets.FashionMNIST(
            root=DATA_ROOT_PATH, train=True,
            download=True, transform=transform)
        train_set, val_set = make_train_val_sets(
            train_val_set, self.validation_set_pc)

        return train_set, val_set, test_set

    def _get_oxford_flowers_sets(self, new_dims, dont_resize):
        flower_data_dir = DATA_ROOT_PATH / Path('flowers')
        train_dir = flower_data_dir / Path('train')
        valid_dir = flower_data_dir / Path('valid')
        test_dir = flower_data_dir / Path('test')

        if not os.path.exists(train_dir):
            extract_to = flower_data_dir
            tar_file = flower_data_dir / Path('flower_data.tar.gz')

            if not os.path.isfile(extract_to):
                print("Downloading flower dataset . . .")
                url = 'https://s3.amazonaws.com/content.' \
                      'udacity-data.com/nd089/flower_data.tar.gz'
                pathlib.Path(flower_data_dir).mkdir(
                    parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, tar_file)

            print("Unzipping . . .")
            with tarfile.open(tar_file, "r:gz") as tar:
                tar.extractall(path=flower_data_dir)
            print("Done")

        training_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            CustomResize(new_dims, dont_resize)
        ])

        validation_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            CustomResize(new_dims, dont_resize)
        ])

        testing_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            CustomResize(new_dims, dont_resize)
        ])

        train_set = torchvision.datasets.ImageFolder(
            train_dir, transform=training_transform)
        test_set = torchvision.datasets.ImageFolder(
            test_dir, transform=testing_transform)
        val_set = torchvision.datasets.ImageFolder(
            valid_dir, transform=validation_transform)

        return train_set, val_set, test_set

    def _get_oxford_pet_sets(self, new_dims, dont_resize):
        train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            CustomResize(new_dims, dont_resize)
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            CustomResize(new_dims, dont_resize)
        ])

        train_val_set = torchvision.datasets.OxfordIIITPet(
            root=DATA_ROOT_PATH, split="trainval",
            download=True, transform=train_transforms)
        test_set = torchvision.datasets.OxfordIIITPet(
            root=DATA_ROOT_PATH, split="test",
            download=True, transform=test_transforms)

        train_set, val_set = make_train_val_sets(
            train_val_set, self.validation_set_pc)

        return train_set, val_set, test_set


def make_train_val_sets(train_val_set, validation_set_pc):
    train_size = int((1 - validation_set_pc) * len(train_val_set))
    val_size = len(train_val_set) - train_size
    train_set, val_set = torch.utils.data.random_split(train_val_set, [train_size, val_size])
    return train_set, val_set


def compute_mean_std(dataset):
    data_r = numpy.dstack([dataset[i][0][:, :, 0] for i in range(len(dataset))])
    data_g = numpy.dstack([dataset[i][0][:, :, 1] for i in range(len(dataset))])
    data_b = numpy.dstack([dataset[i][0][:, :, 2] for i in range(len(dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


if __name__ == "__main__":
    pass
    # datasets = [DatasetName.CIFAR10, DatasetName.CIFAR100,
    #             DatasetName.MNIST, DatasetName.FashionMNIST,
    #             DatasetName.OxfordFlowers, DatasetName.Oxford3Pet]
    # for set in datasets:
    #     print(set)
    #     DataMaker(set, (3, 1000, 1000))

