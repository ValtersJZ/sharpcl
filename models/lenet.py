import torch
from torch import nn

from models.utils import calc_feature_map_size, forward_module_list
from models.classifier import get_trapezoid_classifier

class LeNet5(nn.Module):
    model_type = "LeNet5"
    min_dims = (1, 32, 32)
    dims = (1, 32, 32)

    def __init__(self, image_dim, n_outputs, clf_hidden=1):
        super(LeNet5, self).__init__()

        image_channels = image_dim[0]
        fmap_hw = calc_feature_map_size(
            image_dim[1:], kernel_pool_tuple_ls=[(5, 2), (5, 2), (5, 1)])
        cnn_activation_count = fmap_hw[0] * fmap_hw[1] * 120

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=6, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh()
        )

        # self.classifier = nn.ModuleList([
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(in_features=cnn_activation_count, out_features=84),
        #     nn.Sigmoid(),
        #     nn.Linear(in_features=84, out_features=n_outputs),
        # ])
        self.classifier = get_trapezoid_classifier(ip_size=cnn_activation_count,
                                                   op_size=n_outputs,
                                                   hidden_layers=clf_hidden)

    def forward(self, x):
        h = self.feature_extractor(x)
        y, activation_ls = forward_module_list(h, self.classifier)

        return y, activation_ls


if __name__ == "__main__":
    side, ch = 32, 3
    net = LeNet5(image_dim=(ch, side, side), n_outputs=10)
    x = torch.randn(2, ch, side, side)
    y, act = net(x)
    print(y.size(), end="\n\n")
