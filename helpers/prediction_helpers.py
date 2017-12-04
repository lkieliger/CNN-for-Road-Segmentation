import matplotlib.image as mpimg
import numpy
import tensorflow as tf

from helpers.image_helpers import make_img_overlay, img_crop, concatenate_images, label_to_img
from model import BaselineModel
from program_constants import *


def write_predictions_to_file(predictions, labels, filename):
    """
    Write predictions form a neural network to a file
    
    :param predictions: The predictions made by the neural network
    :param labels: The true class labels 
    :param filename: The name of the file to be written
    
    :return: 
    """
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()


def print_predictions(predictions, labels):
    """
    Print predictions made by a neural network to the console
    
    :param predictions: The predictions made by the neural network
    :param labels: The true class labels
    
    :return: 
    """
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print(str(max_labels) + ' ' + str(max_predictions))


def get_prediction(img, convolutional_model: BaselineModel, tensorflow_session: tf.Session):
    """
    Get the prediction for a given input image
    
    :param convolutional_model: The convolutional neural network model
    :param tensorflow_session: The tensorflow session
    :param img: The image for which to generate the prediction
    
    :return: The prediction
    """
    data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
    data_node = tf.constant(data)
    output = tf.nn.softmax(convolutional_model.model_func()(data_node))
    output_prediction = tensorflow_session.run(output)
    img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

    return img_prediction


def get_prediction_with_groundtruth(filename, image_idx, convolutional_model: BaselineModel, tensorflow_session: tf.Session):
    """
    Get a concatenation of the prediction and groundtruth for a given input file
    
    :param filename: The path to the input image folder
    :param image_idx: The index of the image amongst the dataset
    :param convolutional_model: The convolutional neural network model
    :param tensorflow_session: The tensorflow session
    
    :return: The concatenation of the generated prediction and groundtruth image
    """

    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction(img, convolutional_model, tensorflow_session)
    cimg = concatenate_images(img, img_prediction)

    return cimg


def get_prediction_with_overlay(filename, image_idx, convolutional_model: BaselineModel, tensorflow_session: tf.Session):
    """
    Get the original image with the predictions overlaid on top of it
    
    :param filename: The path to the input image folder
    :param image_idx: The index of the image amongst the dataset
    :param convolutional_model: The convolutional neural network model
    :param tensorflow_session: The tensorflow session
    
    :return: The original image with its predictions overlaid
    """

    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction(img, convolutional_model, tensorflow_session)
    oimg = make_img_overlay(img, img_prediction)

    return oimg
