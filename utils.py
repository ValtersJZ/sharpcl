import os
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from constants import BEST_MODEL_DIR_NAME


def discounting_avg_fn(l_avg, l_batch, n_th):
    return l_avg + (l_batch - l_avg) / n_th


def show_image(img):
    img = img / 2 + 0.5  # Unnormalize.
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def visualize_batch(data_loader, classes, batch_size, model=None, device=None):
    """
        Shows images in batch and ground truth labels. If model and device is
        given, then the prediction are also shown.
    """
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    # Show images and print labels.
    show_image(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    if model is not None:
        if device is None:
            raise Exception("Device parameter required when model is not None.")

        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))


def check_if_best_model(val_metrics, train_metrics, valid_loss_min):
    if val_metrics is None:
        return False
    epoch_val_loss = val_metrics["obj_loss_ep_avg"]
    is_best = epoch_val_loss < valid_loss_min
    return is_best


def save_ckp(state, is_best, ckp_folder_path: Path, model_name: Path, save_only_if_best=False):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    ckp_folder_path.mkdir(parents=True, exist_ok=True)

    model_name = f"{model_name}.pth"

    f_path = ckp_folder_path / model_name
    if not save_only_if_best:
        print("Saving model...")
        torch.save(state, f_path)

    if is_best:
        print("Saving model...")
        best_ckp_folder = ckp_folder_path / BEST_MODEL_DIR_NAME
        best_ckp_folder.mkdir(parents=True, exist_ok=True)

        # remove second best
        for file in os.scandir(best_ckp_folder):
            os.remove(file.path)

        best_fpath = best_ckp_folder / model_name
        if save_only_if_best:
            torch.save(state, best_fpath)
            if not os.path.exists(best_fpath):
                print(f"file FAILED TO SAVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            shutil.copyfile(f_path, best_fpath)


def load_ckp(model_dir_path, model, optimizer, checkpoint_fpath_override=None):
    """
    checkpoint_path: path to load a checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    best_ckp_path = model_dir_path / BEST_MODEL_DIR_NAME
    if checkpoint_fpath_override is None:
        all_files = [f for f in os.listdir(best_ckp_path) if os.path.isfile(best_ckp_path / Path(f))]
        pth_files = [f for f in all_files if ".pth" in f]

        checkpoint_filepath = best_ckp_path / Path(pth_files[0])
    else:
        checkpoint_filepath = checkpoint_fpath_override

    checkpoint = torch.load(checkpoint_filepath)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    val_metrics = checkpoint['val_metrics']
    train_metrics = checkpoint['train_metrics']
    epoch = checkpoint['epoch']
    return model, optimizer, epoch, val_metrics, train_metrics


def define_model_name(epoch, config, train_metrics, val_metrics=None):
    model_name = f"{config['model_type']}"

    if val_metrics is None:
        loss_str = f"val-obj_l___sharp_l___acc___-" \
                   f"train-obj_l{train_metrics['obj_loss_ep_avg']:.3f}" \
                   f"sharp_l{train_metrics['sharp_loss_ep_avg']:.3f}-"
    else:
        loss_str = f"val-obj_l{val_metrics['obj_loss_ep_avg']:.3f}" \
                   f"sharp_l{val_metrics['sharp_loss_ep_avg']:.3f}" \
                   f"acc{val_metrics['accuracy']:.3f}-" \
                   f"train-obj_l{train_metrics['obj_loss_ep_avg']:.3f}" \
                   f"sharp_l{train_metrics['sharp_loss_ep_avg']:.3f}-"

    train_hparam = f"Batch_s{config['batch_size']}"
    opt_str = f"{config['optimizer']['name']}Epoch{epoch}"
    opt_hparams = ''.join([f"{k}{v}" for k, v in
                           config['optimizer']['params'].items()])

    model_name = "".join([model_name, loss_str,
                          train_hparam, opt_str, opt_hparams])
    return model_name


def define_run_name(config):
    model_name = f"{config['model_type']}"

    train_hparam = f"Batch_s{config['batch_size']}"
    opt_str = f"{config['optimizer']['name']}"
    opt_hparams = ''.join([f"{k}{v}" for k, v in
                           config['optimizer']['params'].items()])

    run_name = "".join([model_name, train_hparam, opt_str, opt_hparams])
    return run_name


if __name__ == '__main__':
    pass
