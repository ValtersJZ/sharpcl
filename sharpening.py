import torch


def sharpening_loss_scaler(epoch, batch_idx, batches_in_loader,
                           batch_size, total_epochs, max_coefficient=1):
    overall_batch_number = epoch * batches_in_loader + batch_idx + 1  # 1 ... N
    total_batches = total_epochs * batches_in_loader * batch_size  # 1 ... N * batch_size

    return max_coefficient * (overall_batch_number / total_batches)


def sharpening_loss(activation_tensor_ls, device="cpu", threshold=0.01):
    if activation_tensor_ls == []:
        # print("Activation list empty, returning 0 sharpening looss.")
        return 0

    # activation_tensor_ls = [torch.flatten(a) for a in activation_tensor_ls]
    n_samples = sum([a.size(1) for a in activation_tensor_ls])

    batches = activation_tensor_ls[0].size(0)
    activation_loss_sum = torch.zeros(size=(batches,)).to(device)
    for a in activation_tensor_ls:
        # Loss func for individual activations: y = 4x-4x^2
        a_trans = 4.0 * a - 4.0 * torch.square(a)

        y_of_th = 4.0 * threshold - 4.0 * threshold ** 2
        a_trans = torch.where(a_trans > y_of_th, a_trans, 0)

        activation_loss_sum += torch.sum(a_trans, dim=1)

    sample_loss = activation_loss_sum / n_samples
    total_loss = torch.sum(sample_loss)
    return total_loss
