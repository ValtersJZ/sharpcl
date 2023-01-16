import math

import wandb

sweep_config = {
    'method': 'random'
}

metric = {
    'name': 'loss',
    'goal': 'minimize'
}

sweep_config['metric'] = metric

parameter_dict = {
    'optimizer':
        {
            'values': ['adam', 'sgd']
        },
    # ...
}
sweep_config['parameters'] = parameter_dict
parameter_dict.update({"epochs": {'value': 2}})
parameter_dict.update({
    'learning_rate': {
        'distribution': 'uniform',
        'min': 0.0,
        'max': 0.2,
    },
    'batch_size': {
        'distribution': 'q_log_uniform',
        'q': 1,
        'min': math.log(32),
        'max': math.log(256),
    }
})

# import pprint
# pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="hparam_tuning")