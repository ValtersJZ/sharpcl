import pprint

import wandb

from data import DatasetName
from models import ModelName
from main import main

sweep_config = {
    'method': 'random',
    'metric':
        {'goal': 'minimize', 'name': 'val_loss_ob'},
    'parameters':
        {
            'dataset':
                {'value': DatasetName.MNIST},
            'model_type':
                {'value': ModelName.LeNet5},
            'LOAD_CHECKPOINT': {'value': False},
            "use_sharpening": {'value': True},
            'max_sharpening_k':
                {
                    'distribution': 'uniform',
                    'max': 0.2,
                    'min': 0
                },
            'batch_size':
                {
                    'distribution': 'q_log_uniform_values',
                    'max': 256,
                    'min': 32,
                    'q': 8
                },
            'epochs': {'value': 2},

            'optimizer.name': {'values': ['ADAM', 'SGD']},

            'optimizer.params.lr':
                {
                    'distribution': 'uniform',
                    'max': 0.2,
                    'min': 0
                },
            'optimizer.params.momentum':
                {
                    'distribution': 'uniform',
                    'max': 0.975,
                    'min': 0.7
                },
            'optimizer.params.weight_decay':
                {
                    'distribution': 'log_uniform_values',
                    'max': 0.01,
                    'min': 0.00005
                },

        }
}

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
    wandb.agent(sweep_id, main, count=50)
