import os

import matplotlib.image as mpimg
import numpy

from helpers.image_helpers import img_crop
from program_constants import *


def extract_data(filename, num_images, ):
    """
    Extract the images into a 4D tensor [image index, y, x, channels]
    Values are rescales from [0, 255] down to [-0.5, 0.5]
    
    :param filename: The common path of all images
    :param num_images: The total number of images from which to extract the data
    
    :return: The extracted 4D tensor
    """

    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    # List formed by consecutive series of patches of each image (patches ordered in row order)
    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]

    # List of all the patches, ordered by image
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)


def value_to_class(v):
    """
    Assigns a label to a patch v
    
    :param v: The patch to which to assign a label
    
    :return: The class label in 1-hot format [not road, road]
    """
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

def extract_labels(filename, num_images):
    """
    Extract from ground truth images the class labels and convert them 
    into a 1-hot matrix of the form [image index, label index
    
    :param filename: The common path for all images
    :param num_images: The total number of images for which to extract the labels
    
    :return: A tensor of 1-hot matrix representation of the class labels
    """

    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)

    # List formed by consecutive series of patches of each image (patches ordered in row order)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]

    # List of all the patches, ordered by image
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    # Compute the class label of each patch based on the mean value
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


def split_data(data, labels):

    data_size = data.shape[0]

    if (NUM_IMAGES != TRAINING_SIZE + VALIDATION_SIZE + TEST_SIZE):
        print("NUM_IMAGES: {} SPLIT COUNT: {}".format(NUM_IMAGES, TRAINING_SIZE + VALIDATION_SIZE + TEST_SIZE))
        raise Exception("Dataset split count does not match total number of images!")

    perm_indices = numpy.random.permutation(range(data_size))

    train_bound = round(data_size * TRAINING_SIZE / NUM_IMAGES)
    validation_bound = round(data_size * VALIDATION_SIZE / NUM_IMAGES)

    train_indices = perm_indices[0: train_bound]
    validation_indices = perm_indices[train_bound: train_bound + validation_bound]
    test_indices = perm_indices[train_bound + validation_bound:]

    data_train = data[train_indices, :, :, :]
    data_validation = data[validation_indices, :, :, :]
    data_test = data[test_indices, :, :, :]

    labels_train = labels[train_indices]
    labels_validation = labels[validation_indices]
    labels_test = labels[test_indices]

    return data_train, data_validation, data_test, labels_train, labels_validation, labels_test