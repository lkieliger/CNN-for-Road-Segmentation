import numpy
import tensorflow as tf

def error_rate(predictions, labels):
    """
    Returns the error rate based on dense predictions and 1-hot labels
    
    :param predictions: Prediction in dense format
    :param labels: Class labels in 1-hot format
    
    :return: The error rate, that is the number of time the predictions do not agree with the class labels
    """
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

def accuracy(labels, predictions):
    pass

def f1_score():
    pass