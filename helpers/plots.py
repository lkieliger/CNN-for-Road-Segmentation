import matplotlib.pyplot as plt
import numpy as np
from program_constants import *
import seaborn as sns
sns.set()

def plot_accuracy(data_list, test_accuracy, timestamp, labels=None):

    colors = [sns.color_palette("colorblind")[0],sns.color_palette("colorblind")[2]]

    if labels is None:
        labels = ["Training", "Validation"]

    x = np.array(range(NUM_EPOCHS)) + 1

    for i, data in enumerate(data_list):
        plt.plot(x, data, label=labels[i], color=colors[i])

    plt.plot((1, NUM_EPOCHS), (test_accuracy, test_accuracy), 'k--', color=sns.color_palette("colorblind")[1], label='Test')

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

if __name__ == '__main__':
    import seaborn as sns

    sns.set()

    plt.plot((1, NUM_EPOCHS), (0.2, 0.2), '--', color=sns.color_palette("colorblind")[0],label='Test')
    plt.plot((1, NUM_EPOCHS), (0.4, 0.4), '--', color=sns.color_palette("colorblind")[2], label='Test')
    plt.plot((1, NUM_EPOCHS), (0.6, 0.6), '--', color=sns.color_palette("colorblind")[1], label='Test')

    plt.show()