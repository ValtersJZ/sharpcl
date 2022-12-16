import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_activation_histogram(activations_ls, id, n_bins=100, range=(0.0, 1.0)):
    if activations_ls == []:
        print("Activation list empty, nothing to plot.")
        return 0

    _min, _max = range
    with torch.no_grad():
        # temp visualization --  hist
        a_ls = [a.flatten().to("cpu") for a in activations_ls]
        aa = torch.concat(a_ls, dim=0)
        # _min, _max = torch.min(aa).item(), torch.max(aa).item()
        hist_ = torch.histc(aa, bins=n_bins, min=0, max=1)
        top_freq = torch.max(hist_).item()
        aa = aa.numpy()
        plt.hist(aa, bins=100, range=(0.0, 1.0))
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.title(f"Batch ID:{id}")
        plt.xlim(_min, _max)
        plt.ylim(0, top_freq)
        plt.grid(True)
        plt.show()

    return 1
