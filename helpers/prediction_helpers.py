import matplotlib.image as mpimg
import numpy
import tensorflow as tf

from helpers.image_helpers import make_img_overlay, img_crop, label_to_img
from model import BaselineModel
from program_constants import *


def get_prediction(img, convolutional_model: BaselineModel, tensorflow_session: tf.Session):
    """
    Get the prediction for a given input image
    
    :param convolutional_model: The convolutional neural network model
    :param tensorflow_session: The tensorflow session
    :param img: The image for which to generate the prediction
    
    :return: The prediction
    """
    data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))

    data_indices = range(data.shape[0])
    img_predictions = []

    data_node = tf.placeholder(tf.float32, shape=(None, EFFECTIVE_INPUT_SIZE, EFFECTIVE_INPUT_SIZE, NUM_CHANNELS))
    output = tf.nn.softmax(convolutional_model.model_func()(data_node))

    for i in range(0, data.shape[0], BATCH_SIZE):
        batch_data = data[data_indices[i:i + BATCH_SIZE]]
        output_prediction = tensorflow_session.run(output, feed_dict={data_node: batch_data})
        img_predictions.append(output_prediction)

    stacked_predictions = [numpy.stack(batch_predictions_list) for batch_predictions_list in img_predictions]
    stacked_batches = numpy.vstack(stacked_predictions)

    return label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, stacked_batches)


def get_prediction_with_overlay(filepath, filename, convolutional_model: BaselineModel, tensorflow_session: tf.Session):
    """
    Get the original image with the predictions overlaid on top of it
    
    :param filepath: The path to the input image folder
    :param image_idx: The index of the image amongst the dataset
    :param convolutional_model: The convolutional neural network model
    :param tensorflow_session: The tensorflow session
    
    :return: The original image with its predictions overlaid
    """

    img = mpimg.imread(filepath + filename)
    img_prediction = get_prediction(img, convolutional_model, tensorflow_session)
    oimg = make_img_overlay(img, img_prediction)

    return oimg
