import pprint

import wandb

from data import DatasetName
from models import ModelName
from main import main

from sweep_configs import MNIST_FCNET_sharp_sweep_config

sweep_config = MNIST_FCNET_sharp_sweep_config

if __name__ == '__main__':
    project_name = "fc_mnist_opt_stage1_SHARP"
    # project_name = "dummy"
    sweep_id = wandb.sweep(sweep_config, project=project_name, entity="cl-disco")
    wandb.agent(sweep_id, function=main, count=50)



