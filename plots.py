import matplotlib.pyplot as plt
import numpy as np
import datetime
from program_constants import *

def plot_accuracy(data_list, labels=None):

    if labels is None:
        labels = ["Training", "Validation"]

    x = np.array(range(NUM_EPOCHS)) + 1

    for i, data in enumerate(data_list):
        plt.plot(x, data, label=labels[i])

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("plots/accuracy_{}.png".format(datetime.datetime.now().strftime("%H%M%b%d")))

