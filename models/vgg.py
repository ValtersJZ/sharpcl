'''VGG11/13/16/19 in Pytorch.'''
import math
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models.utils import forward_module_list

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    min_dims = (1, 32, 32)
    dims = (3, 224, 224)

    def __init__(self, vgg_name, image_dim, n_outputs):
        super(VGG, self).__init__()
        self.filepath = Path(f"{vgg_name}.pth")
        self.vgg_name = vgg_name

        self.layers_ls = self._make_layers(cfg[vgg_name], image_channels=image_dim[0])
        self.layers = nn.ModuleList(self.layers_ls)

        cnn_activation_count = self.get_cnn_activation_count(image_dim[1])

        self.classifier_layers = nn.ModuleList([
            nn.Flatten(start_dim=1),
            nn.Linear(cnn_activation_count, 256), nn.Sigmoid(),
            nn.Linear(256, 128), nn.Sigmoid(),
            nn.Linear(128, n_outputs)
        ])

    def forward(self, x):
        x, _ = forward_module_list(x, self.layers_ls)
        x, activations_ls = forward_module_list(x, self.classifier_layers)

        return x, activations_ls

    def _make_layers(self, cfg, image_channels):
        layers = []
        in_channels = image_channels
        for out_channels in cfg:
            if out_channels == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU()]
                in_channels = out_channels
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return layers

    def get_cnn_activation_count(self, image_dim):
        out_channels = cfg[self.vgg_name][-2]
        dim = image_dim
        for el in cfg[self.vgg_name]:
            if el == 'M':
                pool_kernel_size = 2  # strid = kernel_size
                dim = dim // pool_kernel_size
            else:
                kernel_size = 3
                padding = 1
                dim = dim + padding * 2 - kernel_size + 1

        return out_channels * dim * dim


class VGG11(VGG):
    model_type = "VGG11"

    def __init__(self, image_dim, n_outputs):
        super(VGG11, self).__init__("VGG11", image_dim, n_outputs)


class VGG13(VGG):
    model_type = "VGG13"

    def __init__(self, image_dim, n_outputs):
        super(VGG13, self).__init__("VGG13", image_dim, n_outputs)


class VGG16(VGG):
    model_type = "VGG16"

    def __init__(self, image_dim, n_outputs):
        super(VGG16, self).__init__("VGG16", image_dim, n_outputs)


class VGG19(VGG):
    model_type = "VGG19"

    def __init__(self, image_dim, n_outputs):
        super(VGG19, self).__init__("VGG19", image_dim, n_outputs)


class VGG11_Pretrained(nn.Module):
    model_type = "VGG11PretrainedFE"
    min_dims = (1, 32, 32)
    dims = (3, 224, 224)

    def __init__(self, image_dim, n_outputs):
        super().__init__()
        channels = image_dim[0]

        vgg = torchvision.models.vgg11(weights='IMAGENET1K_V1')
        self.features = vgg.features

        if channels != 3:
            print(f"Converting the first layer kernels from 3 channel default to {channels} . . .")
            # self.features[0] = nn.Conv2d(channels, 64, (3, 3), padding=(1, 1))
            self.features[0].in_channels = 1
            if channels == 1:
                w = self.features[0].weight
                self.features[0].weight = torch.nn.Parameter(torch.sum(w, 1, True))
            else:
                raise Exception(f"Network conversion not implemented for {channels} channels!")

        for param in self.features.parameters():
            param.requires_grad = False

        # if channels != 3:
        #     self.features[0] = nn.Conv2d(
        #         channels, 64, (3, 3), padding=(1, 1)
        #     ).requires_grad_(requires_grad=True)

        self.avgpool = vgg.avgpool
        # self.classifier = nn.ModuleList([
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(25088, 2048, bias=True),
        #     nn.Sigmoid(),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(2048, 512, bias=True),
        #     nn.Sigmoid(),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(512, n_outputs, bias=True)
        # ])

        self.classifier = nn.ModuleList([
            nn.Flatten(start_dim=1),
            nn.Linear(25088, 256), nn.Sigmoid(),
            nn.Linear(256, 128), nn.Sigmoid(),
            nn.Linear(128, n_outputs)
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x, activations_ls = forward_module_list(x, self.classifier)

        return x, activations_ls


if __name__ == "__main__":
    pass
    # net = VGG("VGG11", (1, 28, 28), 10)
    # print(net.get_cnn_activation_count(32))
    #
    # inputs = torch.randn((4, 1, 28, 28))
    # outputs, activations_ls = net(inputs)
    # from visualize import plot_activation_histogram
    #
    # plot_activation_histogram(activations_ls, 0)

    # side, ch = 32, 3
    #
    # net = VGG19(image_dim=(ch, side, side), n_outputs=10)
    # x = torch.randn(2, ch, side, side)
    # y, act = net(x)
    # print(y.size(), end="\n\n")

    net = VGG11_Pretrained([3, 32, 32], 10)
    # vgg = torchvision.models.vgg11(pretrained=True)
    vgg = torchvision.models.vgg11(weights='IMAGENET1K_V1')

    # inputs = torch.randn((4, 3, 224, 224))
    # print(vgg)
    print(vgg.features[0].weight.size())
    print(vgg.features[0].bias.size())
    # print(vgg.features(inputs).size())
    # print(vgg.avgpool(vgg.features(inputs)).size())
    # print(vgg(inputs).size())

    vgg = VGG11_Pretrained([1, 30, 30], 10)

    # print(vgg)
    print(vgg.features[0].weight.size())
    print(vgg.features[0].bias.size())

    kernel2d_a = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    kernel2d_b = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
    kernel2d_c = [[3, 3, 3], [3, 3, 3], [3, 3, 3]]

    t = torch.tensor([
        [kernel2d_a, kernel2d_b],
        [kernel2d_a, kernel2d_c]
    ], dtype=torch.float)

    print(t)
    print(t.size())
    print()
    print(torch.mean(t, 1, True))
    print(torch.mean(t, 1, True).size())
