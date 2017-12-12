import os
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg

from helpers.image_helpers import img_float_to_uint8
from helpers.prediction_helpers import get_prediction_with_overlay, get_prediction
from learner import Learner
from program_constants import *


def apply_on_dataset(session, learner, path):
    print("Running prediction on training set")
    prediction_training_dir = "predictions/"

    if not os.path.isdir(prediction_training_dir):
        os.mkdir(prediction_training_dir)

    for j, filename in enumerate(os.listdir(path)):
        i = j+1
        #pimg = get_prediction_with_groundtruth(train_data_filename, i, learner.cNNModel, session)
        #Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
        img = mpimg.imread(path + filename)
        oimg = get_prediction(img, learner.cNNModel, session)
        oimg = img_float_to_uint8( 1 - oimg)
        Image.fromarray(oimg).save(prediction_training_dir + filename)

if __name__ == '__main__':

    learner = Learner()

    # Create a local session to run this computation.
    with tf.Session() as tensorflow_session:
        np.random.seed(SEED)
        tf.set_random_seed(SEED)


        # Restore variables from disk.
        learner.saver.restore(tensorflow_session, "restore/tmp_model.ckpt")
        print("Model restored.")

        apply_on_dataset(tensorflow_session, learner, TEST_DATA_PATH)