import os
import cv2
import matplotlib.image as mpimg
import numpy
import random

from helpers.image_helpers import img_crop
from program_constants import *


def extract_data(path, num_images=-1):
    """
    Extract the images into a 4D tensor [image index, y, x, channels]
    Values are rescales from [0, 255] down to [-0.5, 0.5]
    
    :param path: The common path of all images
    :return: The extracted 4D tensor
    """

    imgs = []
    data = []

    for filename in os.listdir(path):
        imgs.append(cv2.imread(os.path.join(path, filename)))

    IMG_WIDTH = imgs[0].shape[0]

    # Check whether the images are already cropped
    if IMG_WIDTH > EFFECTIVE_INPUT_SIZE:
        # List formed by consecutive series of patches of each image (patches ordered in row order)
        img_patches = [img_crop(i, IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in imgs]

        for i in range(5):
            cv2.imwrite("test_train_patch_"+str(i)+".png", img_patches[0][i])

        # List of all the patches, ordered by image
        data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    else:
        data = imgs


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

def extract_labels(path, num_images=-1):
    """
    Extract from ground truth images the class labels and convert them 
    into a 1-hot matrix of the form [image index, label index
    
    :param path: The common path for all images
    :return: A tensor of 1-hot matrix representation of the class labels
    """

    image_count = num_images if num_images != -1 else len(next(os.walk(path))[2])

    gt_imgs = []
    data = []

    for filename in os.listdir(path):
        gt_imgs.append(cv2.imread(os.path.join(path, filename)))


    IMG_WIDTH = gt_imgs[0].shape[0]

    if IMG_WIDTH > EFFECTIVE_INPUT_SIZE:
        # List formed by consecutive series of patches of each image (patches ordered in row order)
        gt_patches = [img_crop(i, IMG_PATCH_SIZE, IMG_PATCH_SIZE, is_2d=True) for i in gt_imgs]

        for i in range(5):
            cv2.imwrite("test_train_label_"+str(i)+".png", gt_patches[0][i])

        # List of all the patches, ordered by image
        data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    else:
        data = gt_imgs

    # Compute the class label of each patch based on the mean value
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)

def split_patches(data, labels):

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

def balance_dataset(data, labels):
    c0 = 0
    c1 = 0
    for i in range(len(labels)):
        if labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print(len(new_indices))
    print(data.shape)
    data = data[new_indices, :, :, :]
    labels = labels[new_indices]

    data_size = labels.shape[0]
    print(data_size)

    c0 = 0
    c1 = 0
    for i in range(len(labels)):
        if labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    return data, labels
