from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import forward_module_list, calc_feature_map_size


class MiniFCNet(nn.Module):
    model_type = "MiniFCNet"
    min_dims = (1, 1, 1)
    dims = (1, 28, 28)

    def __init__(self, image_dim, n_outputs):
        super().__init__()
        input_len = image_dim[0] * image_dim[1] * image_dim[2]

        self.layers = nn.ModuleList([
            nn.Flatten(start_dim=1),
            nn.Linear(input_len, 120), nn.Sigmoid(),
            nn.Linear(120, 120), nn.Sigmoid(),
            nn.Linear(120, n_outputs)
        ])

    def forward(self, x):
        return forward_module_list(x, self.layers)


if __name__ == "__main__":
    side, ch = 13, 1
    net = MiniFCNet(image_dim=(ch, side, side), n_outputs=10)
    x = torch.randn(2, ch, side, side)
    y, act = net(x)
    print(y.size(), end="\n\n")
