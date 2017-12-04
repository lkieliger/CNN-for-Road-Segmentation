import matplotlib.pyplot as plt
import numpy as np
from program_constants import *

def plot_accuracy(data_list):
    x = np.array(range(NUM_EPOCHS)) + 1

    for data in data_list:
        plt.plot(x, data)

    plt.show()

