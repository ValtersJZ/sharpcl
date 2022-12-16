'''VGG11/13/16/19 in Pytorch.'''
import math
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.utils import forward_module_list

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, image_dim, n_outputs):
        super(VGG, self).__init__()
        self.filepath = Path(f"{vgg_name}.pth")
        self.vgg_name = vgg_name

        self.layers_ls = self._make_layers(cfg[vgg_name], image_channels=image_dim[0])
        self.layers = nn.ModuleList(self.layers_ls)

        cnn_activation_count = self.get_cnn_activation_count(image_dim[1])
        if cnn_activation_count == 0:
            min_width = 32
            width = image_dim[1]
            padding = math.ceil((min_width - width) / 2)
            self.pad = transforms.Pad(padding)

            cnn_activation_count = self.get_cnn_activation_count(min_width)
        else:
            self.pad = False

        self.classifier_layers = nn.ModuleList([
            nn.Flatten(start_dim=1),
            nn.Linear(cnn_activation_count, 256), nn.Sigmoid(),
            nn.Linear(256, 128), nn.Sigmoid(),
            nn.Linear(128, n_outputs)
        ])

    def forward(self, x):
        if self.pad:
            x = self.pad(x)

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


if __name__ == "__main__":
    net = VGG("VGG11", (1, 28, 28), 10)
    print(net.get_cnn_activation_count(32))

    inputs = torch.randn((4, 1, 28, 28))
    outputs, activations_ls = net(inputs)
    from visualize import plot_activation_histogram
    plot_activation_histogram(activations_ls, 0)
