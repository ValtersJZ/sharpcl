import torch
from torch import nn, Tensor

from models.utils import forward_module_list
from models.classifier import get_trapezoid_classifier


class AlexNet(nn.Module):
    model_type = "AlexNet"
    min_dims = (1, 63, 63)
    dims = (3, 224, 224)

    def __init__(self, image_dim, n_outputs, clf_hidden=1) -> None:
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

        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, n_outputs),
        # )
        self.classifier = get_trapezoid_classifier(ip_size=256 * 6 * 6,
                                                   op_size=n_outputs,
                                                   hidden_layers=clf_hidden)

    def forward(self, x: Tensor) -> (Tensor, list):
        h = self.features(x)
        h = self.avgpool(h)
        y, activation_ls = forward_module_list(h, self.classifier)

        return y, activation_ls


if __name__ == "__main__":
    side, ch = 63, 1
    net = AlexNet(image_dim=(ch, side, side), n_outputs=10)
    x = torch.randn(2, ch, side, side)
    y, act = net(x)
    print(y.size(), end="\n\n")
