from models.simple_models import MiniFCNet, BasicCNN1Net, BasicCNN2Net
from models.vgg import VGG


class ModelName:
    MiniFCNet = 'MiniFCNet'
    BasicCNN1Net = 'BasicCNN1Net'
    BasicCNN2Net = 'BasicCNN2Net'
    VGG11 = 'VGG11'

# _model_types = {
#     "MiniFCNet": MiniFCNet,
#     "BasicCNN1Net": BasicCNN1Net,
#     "BasicCNN2Net": BasicCNN2Net,
#     "VGG": VGG
# }
#
#
# def make_model(model_type: str, **kwargs):
#     if model_type not in _model_types.keys():
#         raise Exception(f"Model type '{model_type}' not available."
#                         f"Available models: {sorted(_model_types.keys())}")
#
#     return _model_types[model_type](**kwargs)
#
#
# if __name__ == "__main__":
#     import torch
#     net = make_model("BasicCNN2Net", n_outputs=10) #net = BasicCNN2Net(n_outputs=10)
#     print(type(net).__name__)
#     x = torch.randn(2, 3, 32, 32)
#     y, act = net(x)
#     print(y.size(), end="\n")
