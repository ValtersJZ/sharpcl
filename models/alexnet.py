import math

import torch
from torch import nn, Tensor
from torchvision.transforms import transforms

from models.utils import calc_feature_map_size, forward_module_list


class AlexNet(nn.Module):
    model_type = "AlexNet"
    min_dims = (1, 63, 63)
    dims = (3, 224, 224)

    def __init__(self, image_dim, n_outputs) -> None:
        super(AlexNet, self).__init__()
        in_channels = image_dim[0]

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, n_outputs),
        )

    def forward(self, x: Tensor) -> (Tensor, list):
        return self._forward_impl(x), []

    #  Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


if __name__ == "__main__":
    side, ch = 63, 1
    net = AlexNet(image_dim=(ch, side, side), n_outputs=10)
    x = torch.randn(2, ch, side, side)
    y, act = net(x)
    print(y.size(), end="\n\n")
