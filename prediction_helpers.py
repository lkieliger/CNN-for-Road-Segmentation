import numpy

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