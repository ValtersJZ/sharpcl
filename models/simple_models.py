from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import forward_module_list, calc_feature_map_size


class MiniFCNet(nn.Module):
    model_type = "MiniFCNet"
    min_dims = (1, 1, 1)
    dims = (1, 28, 28)

    def __init__(self, image_dim, n_outputs, hidden_layer_widths=None):
        super().__init__()
        print(hidden_layer_widths)
        use_3rd_hlayer = hidden_layer_widths["3"]["used"]
        hidden_layer_widths = [
            hidden_layer_widths["1"]["value"],
            hidden_layer_widths["2"]["value"],
            hidden_layer_widths["3"]["value"]]

        if not use_3rd_hlayer:
            hidden_layer_widths = hidden_layer_widths[:-1]

        print(f"hidden_layer_widths: {hidden_layer_widths}")

        input_len = image_dim[0] * image_dim[1] * image_dim[2]
        hidden_layer_widths = [400, 400] if hidden_layer_widths is None else hidden_layer_widths
        assert self.is_valid_hidden_layer_widths(hidden_layer_widths)

        hidden_layer_widths = [width for width in hidden_layer_widths if width is not None]
        layer_widths = [input_len] + hidden_layer_widths + [n_outputs]

        layer_pairs = [[nn.Linear(layer_widths[l], layer_widths[l + 1]), nn.Sigmoid()] for l in
                       range(len(layer_widths) - 1)]
        layers = [l for pair in layer_pairs for l in pair][:-1]
        print(layers)

        self.layers = nn.ModuleList(
            [nn.Flatten(start_dim=1)] + layers
        )
        # self.layers = nn.ModuleList([
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(input_len, 400), nn.Sigmoid(),
        #     nn.Linear(400, 400), nn.Sigmoid(),
        #     nn.Linear(400, n_outputs)
        # ])

    def forward(self, x):
        return forward_module_list(x, self.layers)

    @staticmethod
    def is_valid_hidden_layer_widths(lst):
        return not any(lst[i] is None and i < len(lst) - 1 and lst[i + 1] is not None for i in range(len(lst)))


if __name__ == "__main__":
    side, ch = 13, 1
    net = MiniFCNet(image_dim=(ch, side, side), n_outputs=10)
    x = torch.randn(2, ch, side, side)
    y, act = net(x)
    print(y.size(), end="\n\n")
