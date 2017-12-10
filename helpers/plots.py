import matplotlib.pyplot as plt
import numpy as np
from program_constants import *

def plot_accuracy(data_list, timestamp, labels=None):

    if labels is None:
        labels = ["Training", "Validation"]

    x = np.array(range(NUM_EPOCHS)) + 1

    for i, data in enumerate(data_list):
        plt.plot(x, data, label=labels[i])

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("plots/accuracy_{}.png".format(timestamp))

def plot_conv_weights(weights, timestamp):

    num_filters = weights.shape[3]
    num_grids = int(np.ceil(np.sqrt(num_filters)))

    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            data = weights[:, :, :, i]
            w_min = np.min(data)
            w_max = np.max(data)
            img = (data - w_min) / (w_max - w_min)

            # Plot image.
            ax.imshow(img,interpolation='nearest')

        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig("plots/weights_{}.png".format(timestamp))
