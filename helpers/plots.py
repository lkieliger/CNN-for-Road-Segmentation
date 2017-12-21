import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from program_constants import *


def plot_accuracy(data_list, test_accuracy, timestamp, labels=None):
    if labels is None:
        labels = ["Training", "Validation"]

    x = np.array(range(NUM_EPOCHS)) + 1

    for i, data in enumerate(data_list):
        plt.plot(x, data, label=labels[i])

    plt.plot((1, NUM_EPOCHS), (test_accuracy, test_accuracy), 'k--', label='Test')

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    if not os.path.isdir("plots/"):
        os.mkdir("plots/")
    plt.savefig("plots/accuracy_{}.png".format(timestamp))


def plot_conv_weights(weights, timestamp, fileprefix=""):
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
            ax.imshow(img, interpolation='nearest')

        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(fileprefix + "plots/weights_{}.png".format(timestamp))


def plot_from_csv(data1, data2):
    df_train = pd.read_csv(data1[0])
    df_val = pd.read_csv(data1[1])

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4), gridspec_kw={'width_ratios': [1, 1]})
    ax1.plot(df_train.index, df_train['# accuracy'], label="Train")
    ax1.plot(df_val.index, df_val['# accuracy'], label="Validation")
    ax1.plot((1, 100), (data1[2], data1[2]), label="Test")
    ax1.tick_params(right=True, direction='out')
    ax1.set_title("L2-Regularization")
    ax1.yaxis.grid(linestyle=':')

    df_train = pd.read_csv(data2[0])
    df_val = pd.read_csv(data2[1])

    ax2.plot(df_train.index, df_train['# accuracy'], label="Train")
    ax2.plot(df_val.index, df_val['# accuracy'], label="Validation")
    ax2.plot((1, 100), (data2[2], data2[2]), label="Test")
    ax2.yaxis.grid(linestyle=':')
    ax2.set_title("Dropout Regularization")
    ax2.tick_params(labelright=True, right=True)
    ax2.legend(loc="lower right")
    f.text(0.5, 0.04, 'Epochs', ha='center')
    f.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')

    plt.show()


if __name__ == '__main__':
    plot_from_csv(["../logs/11-21_20_09/train_scores.csv", "../logs/11-21_20_09/val_scores.csv", 0.87648],
                  ["../logs/11-20_15_55/train_scores.csv", "../logs/11-20_15_55/val_scores.csv", 0.9109])
