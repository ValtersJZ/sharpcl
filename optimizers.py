from torch import optim


def get_optimizer(optimizer_name, model_params, **opt_params):
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model_params, **opt_params)
    elif optimizer_name == "ADAM":
        optimizer = optim.Adam(model_params, **opt_params)
    else:
        raise Exception(f"Optimizer {optimizer_name} not recognized.")
    return optimizer
