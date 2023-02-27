import pprint

import wandb

from data import DatasetName
from models import ModelName
from main import main

from sweep_configs import MNIST_FCNET_sharp_sweep_config

sweep_config = MNIST_FCNET_sharp_sweep_config

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project="fc_mnist_opt_stage1_SHARP")
    wandb.agent(sweep_id, main, count=50)
