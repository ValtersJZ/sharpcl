from models.simple_models import MiniFCNet, BasicCNN1Net, BasicCNN2Net
from models.vgg import VGG11, VGG13, VGG16, VGG19, VGG11_Pretrained
from models.lenet import LeNet5
from models.alexnet import AlexNet


class ModelName:
    MiniFCNet = MiniFCNet.model_type
    BasicCNN1Net = BasicCNN1Net.model_type
    BasicCNN2Net = BasicCNN2Net.model_type
    VGG11 = VGG11.model_type
    VGG13 = VGG13.model_type
    VGG16 = VGG16.model_type
    VGG19 = VGG19.model_type
    VGG11_PTR = VGG11_Pretrained.model_type
    LeNet5 = LeNet5.model_type
    AlexNet = AlexNet.model_type


_MODEL_TYPES = {
    ModelName.MiniFCNet: MiniFCNet,
    ModelName.BasicCNN1Net: BasicCNN1Net,
    ModelName.BasicCNN2Net: BasicCNN2Net,
    ModelName.VGG11: VGG11,
    ModelName.VGG13: VGG13,
    ModelName.VGG16: VGG16,
    ModelName.VGG19: VGG19,
    ModelName.VGG11_PTR: VGG11_Pretrained,
    ModelName.LeNet5: LeNet5,
    ModelName.AlexNet: AlexNet
}

MODEL_MIN_DIMS = {model.model_type: model.min_dims for
                  model in _MODEL_TYPES.values()}

MODEL_DIMS = {model.model_type: model.dims for
              model in _MODEL_TYPES.values()}


def get_model(model_type, **kwargs):
    if model_type not in _MODEL_TYPES.keys():
        raise Exception(f"Model type '{model_type}' not available."
                        f"Available models: {sorted(_MODEL_TYPES.keys())}")

    model = _MODEL_TYPES[model_type](**kwargs)
    return model


if __name__ == "__main__":
    import torch

    net = get_model("VGG11", image_dim=(3, 32, 32), n_outputs=10)
    print(type(net).__name__)
    x = torch.randn(2, 3, 32, 32)
    y, act = net(x)
    print(y.size(), end="\n")
