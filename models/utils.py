import torch
import torch.nn as nn


def forward_module_list(x, layers, activation_to_collect=nn.Sigmoid):
    activations_ls = []

    for layer in layers:
        # print(f"layer: {layer}")
        # print(f"pre  x.size(): {x.size()}")
        x = layer(x)
        if isinstance(layer, activation_to_collect):
            activations_ls.append(torch.flatten(x, 1))
    return x, activations_ls


def calc_feature_map_size(ip_dim, kernel_pool_tuple_ls):
    feature_map_dim = list(ip_dim)
    for i, dim in enumerate(ip_dim):
        for kernel_size, pool_factor in kernel_pool_tuple_ls:
            dim = (dim - kernel_size + 1) // pool_factor
        feature_map_dim[i] = dim
    return tuple(feature_map_dim)


if __name__ == "__main__":
    pass
    # print(calc_feature_map_size((32, 32), [(5,2), (5,2)]))
