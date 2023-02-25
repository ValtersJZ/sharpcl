import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import wandb

from constants import MODEL_PATH, BEST_MODEL_DIR_NAME
from data import DatasetName, DataMaker
from models import ModelName, get_model, MODEL_DIMS

from optimizers import get_optimizer
from sharpening import sharpening_loss_scaler, sharpening_loss

from state import State
from utils import load_ckp, check_if_best_model, define_model_name, save_ckp, discounting_avg_fn, define_run_name

use_wandb = True

TRAIN_MODEL = True
LOAD_CHECKPOINT = True

config_defaults = {
    "dataset": DatasetName.MNIST,
    "batch_size": 256,
    "epochs": 5,
    "model_type": ModelName.MiniFCNet,
    "model_params": {
        "hidden_layer_widths": {
            "1": {"value": 400},
            "2": {"value": 400},
            "3": {"value": 20,
                  "used": False}
        }
    },
    "run_name": "Not_sharp",
    "optimizer": {
        "name": "SGD",
        "params": {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 0.0001
        }
    },
    "use_sharpening": False,
    # "max_sharpening_k": 0.1,
    "TRAIN_MODEL": TRAIN_MODEL,
    "LOAD_CHECKPOINT": LOAD_CHECKPOINT
}


def main(config=config_defaults):
    if use_wandb:
        wandb.init(project="wandb-FC_MNIST_opt1", entity="cl-disco", config=config, name=define_run_name(config))
        config = wandb.config

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"device: {device}")
    print(f"dataset: {config['dataset']}, model: {config['model_type']}")

    # Data.
    # model_min_ip_dims = MODEL_MIN_DIMS[config["model_type"]]
    model_std_ip_dims = MODEL_DIMS[config["model_type"]]

    data_maker = DataMaker(config["dataset"], model_std_ip_dims)
    image_dim = data_maker.get_img_dims()
    n_outputs = data_maker.get_num_outputs()
    classes = data_maker.get_classes()

    train_loader, val_loader, test_loader = data_maker.make_loaders(
        config["batch_size"])

    # Model.
    net = get_model(config["model_type"],
                    image_dim=image_dim, n_outputs=n_outputs, **config["model_params"])
    if use_wandb:
        wandb.watch(net)
    net.to(device)

    model_dir_path = MODEL_PATH / Path(config["dataset"]) / Path(
        config["model_type"]) / config["run_name"]

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        config["optimizer"]["name"],
        net.parameters(),
        **config["optimizer"]["params"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"])

    # add warmup, cool down, cyclic cosine scheduler, noise?, random restart?, ..?
    state = State()

    if config['LOAD_CHECKPOINT']:
        if os.path.exists(model_dir_path / BEST_MODEL_DIR_NAME):
            model, optimizer, epoch_start, val_metrics, train_metrics = load_ckp(
                model_dir_path, net, optimizer)

            state.update_state(epoch_start=epoch_start)
            if val_metrics is not None:
                state.update_state(valid_obj_l_min=val_metrics["obj_loss_ep_avg"])
        else:
            print("Proceeding to train a new model.")

    if TRAIN_MODEL:
        train(net, device, train_loader, val_loader, optimizer, criterion, scheduler,
              model_dir_path, state, config)

    test(net, device, test_loader, classes, criterion, config)


def train(net, device, train_loader, val_loader, optimizer,
          criterion, scheduler, model_ckp_dir_path, state, config):
    start_epoch = state.epoch_start
    stop_epoch = start_epoch + config["epochs"]
    best_state = None
    best_name = None
    for epoch in range(start_epoch, stop_epoch):
        train_metrics = train_one_epoch(epoch, net, device, train_loader,
                                        optimizer, criterion, start_epoch, config)
        if val_loader is not None:
            val_metrics = validate(net, device, val_loader, criterion, config)
        else:
            val_metrics = None
        current_state = {
            'epoch': epoch + 1,
            'val_metrics': val_metrics,
            'train_metrics': train_metrics,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if best_state is None:
            best_state = current_state

        is_best = check_if_best_model(val_metrics, train_metrics,
                                      state.valid_obj_l_min)
        model_name = define_model_name(epoch, config,
                                       train_metrics, val_metrics)

        if is_best:
            best_state = current_state
            best_name = model_name

        scheduler.step()
        if val_loader is not None:
            state.update_state(valid_obj_l_min=val_metrics["obj_loss_ep_avg"])

    save_ckp(best_state, True, model_ckp_dir_path,
             best_name, save_only_if_best=True)


def train_one_epoch(epoch, net, device, train_loader,
                    optimizer, criterion, start_epoch, config):
    log_running_loss = 0.0
    metrics_train_epoch = {"obj_loss_ep_avg": 0, "sharp_loss_ep_avg": 0}
    net.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs, activations_ls = net(inputs)

        # Loss
        if not config["use_sharpening"]:
            sharpening_scaler = 0
        else:
            batches_in_loader = len(train_loader)
            sharpening_scaler = sharpening_loss_scaler(
                epoch - start_epoch, batch_idx, batches_in_loader,
                config["batch_size"], config["epochs"],
                max_coefficient=config["max_sharpening_k"]
            )

        objective_loss = criterion(outputs, labels)
        sharp_loss = sharpening_loss(activations_ls, device)
        loss = objective_loss + sharpening_scaler * sharp_loss

        if use_wandb:
            wandb.log({
                "loss_ob": objective_loss,
                "loss_sharp": sharp_loss,
                "sharpening_scaler": sharpening_scaler,
                "loss": loss
            })

        loss.backward()
        optimizer.step()

        nth_batch = batch_idx + 1
        metrics_train_epoch["obj_loss_ep_avg"] = discounting_avg_fn(
            metrics_train_epoch["obj_loss_ep_avg"], objective_loss, nth_batch)
        metrics_train_epoch["sharp_loss_ep_avg"] = discounting_avg_fn(
            metrics_train_epoch["sharp_loss_ep_avg"], sharp_loss, nth_batch)

        # log running loss
        log_running_loss += loss
        log_period = 300
        if batch_idx % log_period == (log_period - 1):  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {log_running_loss / (log_period - 1):.3f}'
                  f' objective loss: {objective_loss :.3f}'
                  f' sharpening loss: {sharp_loss :.3f}')
            log_running_loss = 0.0
            # plot_activation_histogram(activations_ls, epoch)

    return metrics_train_epoch


def validate(net, device, val_loader, criterion, config):
    metrics_val = {"obj_loss_ep_avg": np.inf, "sharp_loss_ep_avg": np.inf, "accuracy": 0.0}
    total, correct = 0, 0
    objective_loss_ls, sharp_loss_ls = [], []

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, activations_ls = net(inputs)
            _, preds = torch.max(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += len(preds)

            objective_loss_ls.append(criterion(outputs, labels))
            sharp_loss_ls.append(sharpening_loss(activations_ls, device))

        batches_in_loader = len(val_loader)
        sample_weights = torch.ones(batches_in_loader)
        sample_weights[-1] = batches_in_loader * config["batch_size"] % config["batch_size"]
        sample_weight_norm_l1 = F.normalize(sample_weights, dim=0, p=1)

        objective_loss = torch.dot(torch.Tensor(objective_loss_ls), sample_weight_norm_l1).item()
        sharp_loss = torch.dot(torch.Tensor(sharp_loss_ls), sample_weight_norm_l1).item()

        metrics_val["obj_loss_ep_avg"] = objective_loss
        metrics_val["sharp_loss_ep_avg"] = sharp_loss
        metrics_val["accuracy"] = val_accuracy = correct / total

        if use_wandb:
            wandb.log({
                "val_loss_ob": objective_loss,
                "val_loss_sharp": sharp_loss,
                "val_accuracy": val_accuracy
            })

    print(f"val: {correct} / {total} = {correct / total : 0.3f}")
    return metrics_val


def test(net, device, test_loader, classes, criterion, config):
    net.eval()
    # Test
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    all_obj_losses = []
    all_sharp_losses = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs, activations_ls = net(images)

            all_obj_losses.append(criterion(outputs, labels))
            all_sharp_losses.append(sharpening_loss(activations_ls, device))

            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {str(classname):5s} is {accuracy:.1f} %')

    total = sum([count for count in total_pred.values()])
    correct = sum([count for count in correct_pred.values()])
    test_accuracy = correct / total
    print(f'Accuracy of the network on the {total} test images: {100 * test_accuracy: .3f} %')

    batches_in_loader = len(test_loader)
    sample_weights = torch.ones(batches_in_loader)
    sample_weights[-1] = batches_in_loader * config["batch_size"] % config["batch_size"]
    sample_weight_norm_l1 = F.normalize(sample_weights, dim=0, p=1)

    objective_loss = torch.dot(torch.Tensor(all_obj_losses), sample_weight_norm_l1).item()
    sharp_loss = torch.dot(torch.Tensor(all_sharp_losses), sample_weight_norm_l1).item()

    if use_wandb:
        wandb.log({
            "test_accuracy": test_accuracy,
            "test_loss_ob": objective_loss,
            "test_loss_sharp": sharp_loss
        })


if __name__ == '__main__':
    torch.set_printoptions(linewidth=200)
    main(config=config_defaults)
