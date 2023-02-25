import math

import torch.nn as nn


def ceil_to_nearest_even(num):
    ceil = math.ceil(num / 2.0) * 2
    return ceil


def get_trapezoid_classifier(ip_size, op_size, hidden_layers, output_type="nn.ModuleList"):
    percent = [k / (hidden_layers + 1) for k in range(1, hidden_layers + 1)]
    hidden_layers_sizes_float = [ip_size * pc + op_size * (1 - pc) for pc in percent]
    hidden_layers_sizes = [math.ceil(h) for h in hidden_layers_sizes_float]
    layers_sizes = [ip_size] + hidden_layers_sizes + [op_size]

    prev_layer = None
    layer_ls = [nn.Flatten(start_dim=1)]
    for layer in layers_sizes:
        if prev_layer is None:
            prev_layer = layer
            continue

        layer_ls.append(nn.Linear(prev_layer, layer))
        layer_ls.append(nn.Sigmoid())

        prev_layer = layer

    layer_ls = layer_ls[:-1]  # Remove final activation function.

    if output_type == "nn.ModuleList":
        return nn.ModuleList(layer_ls)
    elif output_type == "nn.Sequential":
        return nn.Sequential(*layer_ls)
    else:
        raise Exception("Output type not recognized!")


if __name__ == "__main__":
    print(get_trapezoid_classifier(100, 10, 1))
    print(get_trapezoid_classifier(100, 10, 2))
    print(get_trapezoid_classifier(100, 10, 3))
