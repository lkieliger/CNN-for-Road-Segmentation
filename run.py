import os

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from PIL import Image

from helpers.image_helpers import img_float_to_uint8
from helpers.prediction_helpers import get_prediction
from learner import Learner
from program_constants import *
from utils.mask_to_submission import masks_to_submission


def apply_on_dataset(session, learner, path, model_name):
    print("Running prediction on training set")
    prediction_training_dir = "predictions/"

    if not os.path.isdir(prediction_training_dir):
        os.mkdir(prediction_training_dir)

    res_file_name = []
    for j, filename in enumerate(os.listdir(path)):
        i = j + 1
        img = mpimg.imread(path + filename)
        oimg = get_prediction(img, learner.cNNModel, session)
        oimg = img_float_to_uint8(1 - oimg)
        Image.fromarray(oimg).save(prediction_training_dir + filename)
        res_file_name.append(prediction_training_dir + filename)
        print(prediction_training_dir + filename + " is saved")
    masks_to_submission("submissions/" + model_name + "_submission.csv", *res_file_name)


if __name__ == '__main__':
    learner = Learner()
    model_name = "cnnb-full-175"

    # Create a local session to run this computation.
    with tf.Session() as tensorflow_session:
        np.random.seed(SEED)
        tf.set_random_seed(SEED)

        # Restore variables from disk.
        learner.saver.restore(tensorflow_session, "restore/" + model_name + "_model.ckpt")
        print("Model restored.")
        apply_on_dataset(tensorflow_session, learner, TEST_DATA_PATH, model_name)
