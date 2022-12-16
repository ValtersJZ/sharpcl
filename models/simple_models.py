from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import forward_module_list, calc_feature_map_size


class MiniFCNet(nn.Module):
    model_type = "MiniFCNet"

    def __init__(self, image_dim, n_outputs, name="fc_net"):
        super().__init__()
        self.filepath = Path(f"{name}.pth")
        input_len = image_dim[0] * image_dim[1] * image_dim[2]

        self.layers = nn.ModuleList([
            nn.Flatten(start_dim=1),
            nn.Linear(input_len, 120), nn.Sigmoid(),
            nn.Linear(120, 120), nn.Sigmoid(),
            nn.Linear(120, n_outputs)
        ])

    def forward(self, x):
        return forward_module_list(x, self.layers)


class BasicCNN1Net(nn.Module):
    model_type = "BasicCNN1Net"

    def __init__(self, image_dim, n_outputs, name="cnn_net"):
        super().__init__()
        self.filepath = Path(f"{name}.pth")

        image_channels = image_dim[0]
        fmap_dim = calc_feature_map_size(ip_dim=image_dim[1:],
                                         kernel_pool_tuple_ls=[(5, 2), (5, 2)])
        self.layers = nn.ModuleList([
            nn.Conv2d(image_channels, 6, 5), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.Conv2d(6, 16, 5), nn.MaxPool2d(2, 2), nn.ReLU(),

            nn.Flatten(start_dim=1),
            nn.Linear(16 * fmap_dim[0] * fmap_dim[1], 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, n_outputs),
        ])

    def forward(self, x):
        return forward_module_list(x, self.layers)


class BasicCNN2Net(nn.Module):
    model_type = "BasicCNN2Net"

    def __init__(self, image_dim, n_outputs, name="wb_cnn_net"):
        super(BasicCNN2Net, self).__init__()
        self.filepath = Path(f"{name}.pth")

        image_channels = image_dim[0]
        fmap_dim = calc_feature_map_size(ip_dim=image_dim[1:],
                                         kernel_pool_tuple_ls=[(4, 2), (4, 2)])
        self.layers = nn.ModuleList([
            nn.Conv2d(image_channels, 32, (4, 4)), nn.MaxPool2d(kernel_size=(2, 2)), nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4)), nn.MaxPool2d((2, 2)), nn.ReLU(),

            nn.Flatten(start_dim=1),
            nn.Linear(64 * fmap_dim[0] * fmap_dim[1], 512), nn.ReLU(),
            nn.Linear(512, 256), nn.Sigmoid(),
            nn.Linear(256, 128), nn.Sigmoid(),
            nn.Linear(128, 10),
        ])

    def forward(self, x):
        return forward_module_list(x, self.layers)


def test():
    net = BasicCNN2Net(image_dim=(3, 32, 32), n_outputs=10)
    x = torch.randn(2, 3, 32, 32)
    y, act = net(x)
    print(y.size(), end="\n\n")
