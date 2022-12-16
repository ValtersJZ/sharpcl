import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data import get_dataset, get_dataset_info, DatasetName
from sharpening import sharpening_loss_scaler, sharpening_loss
from constants import MODEL_PATH, BEST_MODEL_DIR_NAME
from models import ModelName, MiniFCNet, BasicCNN1Net, BasicCNN2Net, VGG
from utils import save_ckp, load_ckp, define_mode_name, check_if_best_model

# wandb.init(project="cl-disco", entity="valtersjz", reinit=True)
from visualize import plot_activation_histogram

config = {
    "dataset": DatasetName.FashionMNIST,
    "batch_size": 128,
    "epochs": 3,
    "model_type": ModelName.MiniFCNet,
    "run_name": "run_delete_me_too_sharp",
    "optimizer": {
        "name": "SGD",
        "params": {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 0.0001
        }
    },
    "use_sharpening": True,
    "max_sharpening_k": 0.2
    # "optimizer": {
    #     "name": "ADAM",
    #     "params": {
    #     }
    # }
}

TRAIN_MODEL = True
LOAD_CHECKPOINT = True


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"device: {device}")
    print(f"dataset: {config['dataset']}, model: {config['model_type']}")

    train_set, val_set, test_set = get_dataset(dataset=config["dataset"],
                                               validation_set_pc=0.1)
    image_dim, n_outputs, classes = get_dataset_info(dataset=config["dataset"])

    train_loader = DataLoader(train_set, batch_size=config["batch_size"],
                              shuffle=True, num_workers=3)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"],
                            shuffle=True, num_workers=3)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"],
                             shuffle=False, num_workers=3)

    # Model.
    model_type = config["model_type"]
    if model_type == "MiniFCNet":
        net = MiniFCNet(image_dim=image_dim, n_outputs=n_outputs)
    elif model_type == "BasicCNN2Net":
        net = BasicCNN2Net(image_dim, n_outputs)
    elif model_type == "BasicCNN1Net":
        net = BasicCNN1Net(image_dim, n_outputs)
    elif model_type == "VGG11":
        net = VGG("VGG11", image_dim, n_outputs=n_outputs)
    else:
        raise Exception(f"Model {model_type} not recognized.")

    net.to(device)
    model_dir_path = MODEL_PATH / Path(config["dataset"]) / Path(config["model_type"]) / config["run_name"]

    criterion = nn.CrossEntropyLoss()

    if config["optimizer"]["name"] == "SGD":
        optimizer = optim.SGD(net.parameters(), **config["optimizer"]["params"])
    elif config["optimizer"]["name"] == "ADAM":
        optimizer = optim.Adam(net.parameters(), **config["optimizer"]["params"])
    else:
        raise Exception(f"Optimizer {config['optimizer']['name']} not recognized.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"])
    # add warmup, cool down, cyclic cosine scheduler, noise?, random restart?, ..?
    epoch_start = 0
    valid_obj_l_min = np.inf

    if LOAD_CHECKPOINT:
        if os.path.exists(model_dir_path / BEST_MODEL_DIR_NAME):
            model, optimizer, epoch_start, val_metrics, train_metrics = load_ckp(
                model_dir_path, net, optimizer)

            if val_metrics is not None:
                valid_obj_l_min = val_metrics["obj_loss_ep_avg"]
        else:
            print("No checkpoint exists, proceeding to train a new model.")

    if TRAIN_MODEL:
        train(net, device, train_loader, optimizer, criterion, scheduler,
              model_dir_path, val_loader, epoch_start, valid_obj_l_min)

    test(net, device, test_loader, classes)


def train(net, device, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
          criterion, scheduler, model_ckp_dir_path, val_loader=None, epoch_start=0,
          valid_obj_l_min_input=np.inf):
    valid_obj_l_min = valid_obj_l_min_input

    for epoch in range(epoch_start, epoch_start + config["epochs"]):
        train_metrics = train_epoch(epoch, net, device, train_loader,
                                    optimizer, criterion, epoch_start=epoch_start)
        if val_loader is not None:
            val_metrics = validate(net, device, val_loader, criterion)
        else:
            val_metrics = None

        state = {
            'epoch': epoch + 1,
            'val_metrics': val_metrics,
            'train_metrics': train_metrics,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        is_best = check_if_best_model(val_metrics, train_metrics, valid_obj_l_min)
        model_name = define_mode_name(epoch, config, train_metrics, val_metrics)
        save_ckp(state, is_best, model_ckp_dir_path, model_name, save_only_if_best=True)

        scheduler.step()

        if val_loader is not None:
            valid_obj_l_min = val_metrics["obj_loss_ep_avg"]


def train_epoch(epoch, net, device, train_loader: DataLoader,
                optimizer: torch.optim.Optimizer, criterion, epoch_start=0):
    log_running_loss = 0.0
    metrics_train_epoch = {"obj_loss_ep_avg": 0, "sharp_loss_ep_avg": 0}

    net.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs, activations_ls = net(inputs)

        # Loss
        objective_loss = criterion(outputs, labels)

        batches_in_loader = len(train_loader)
        if config["use_sharpening"]:
            sharpening_scaler = sharpening_loss_scaler(
                epoch - epoch_start, batch_idx, batches_in_loader,
                config["batch_size"], config["epochs"],
                max_coefficient=config["max_sharpening_k"]
            )
        else:
            sharpening_scaler = 0

        sharp_loss = sharpening_loss(activations_ls, device)
        loss = objective_loss + sharpening_scaler * sharp_loss

        loss.backward()
        optimizer.step()

        def discounting_avg_fn(l_avg, l_batch, n_th):
            return l_avg + (l_batch - l_avg) / n_th

        nth_batch = batch_idx + 1
        metrics_train_epoch["obj_loss_ep_avg"] = discounting_avg_fn(metrics_train_epoch["obj_loss_ep_avg"],
                                                                    objective_loss, nth_batch)
        metrics_train_epoch["sharp_loss_ep_avg"] = discounting_avg_fn(metrics_train_epoch["sharp_loss_ep_avg"],
                                                                      sharp_loss, nth_batch)

        # log running loss
        log_running_loss += loss
        log_period = 300
        if batch_idx % log_period == (log_period - 1):  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {log_running_loss / (log_period - 1):.3f}'
                  f' objective loss: {objective_loss :.3f}'
                  f' sharpening loss: {sharp_loss :.3f}')
            log_running_loss = 0.0
            print(f"loss = objective_loss + sharpening_scaler * sharp_loss ="
                  f" {loss} = {objective_loss} + {sharpening_scaler}  * {sharp_loss}="
                  f"{objective_loss} + {sharpening_scaler * sharp_loss}")
            plot_activation_histogram(activations_ls, epoch)

    return metrics_train_epoch


def validate(net, device, val_loader, criterion):
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
        metrics_val["accuracy"] = correct / total

    print(f"val: {correct} / {total} = {correct / total : 0.3f}")
    return metrics_val


def test(net, device, test_loader: DataLoader, classes):
    net.eval()
    # Test
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs, _ = net(images)
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
    print(f'Accuracy of the network on the {total} test images: {100 * correct / total: .3f} %')


if __name__ == '__main__':
    torch.set_printoptions(linewidth=200)
    main()
