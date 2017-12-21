import numpy


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


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def accuracy(tp, fp, tn, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def f1_score(tp, fp, fn):
    pre = precision(tp, fp)
    rec = recall(tp, fn)

    return (2.0 * pre * rec) / (pre + rec)
