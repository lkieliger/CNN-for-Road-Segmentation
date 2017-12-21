import os

import matplotlib.image as mpimg
import numpy

from helpers.image_helpers import img_crop
from program_constants import *


def extract_all_data(path, num_images=-1):
    """
    Extract the images into a 4D tensor [image index, y, x, channels]
    Values are rescales from [0, 255] down to [-0.5, 0.5]

    :param path: The common path of all images
    :return: The extracted 4D tensor
    """

    imgs = []
    data = []
    print("Extracting all patches from " + path)

    for filename in os.listdir(path):
        print(filename)
        imgs.append(mpimg.imread(os.path.join(path, filename)))

    data = []
    if (len(imgs) > 0):
        IMG_WIDTH = imgs[0].shape[0]

        # Check whether the images are already cropped
        if IMG_WIDTH > EFFECTIVE_INPUT_SIZE:
            # List formed by consecutive series of patches of each image (patches ordered in row order)
            img_patches = [img_crop(i, IMG_PATCH_SIZE, IMG_PATCH_SIZE, is_2d=False) for i in imgs]

            # List of all the patches, ordered by image
            data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
        else:
            data = imgs
            print("Detected already cropped image")

    return numpy.asarray(data)


def extract_all_labels(path, num_images=-1, convert_to_1hot=True):
    """
    Extract from ground truth images the class labels and convert them 
    into a 1-hot matrix of the form [image index, label index

    :param path: The common path for all images
    :return: A tensor of 1-hot matrix representation of the class labels
    """
    print("Extracting all labels from " + path)
    image_count = num_images if num_images != -1 else len(next(os.walk(path))[2])

    gt_imgs = []
    data = []

    for filename in os.listdir(path):
        print(filename)
        gt_imgs.append(mpimg.imread(os.path.join(path, filename)))

    data = []
    if (len(gt_imgs) > 0):
        IMG_WIDTH = gt_imgs[0].shape[0]

        if IMG_WIDTH > EFFECTIVE_INPUT_SIZE:
            # List formed by consecutive series of patches of each image (patches ordered in row order)
            gt_patches = [img_crop(i, IMG_PATCH_SIZE, IMG_PATCH_SIZE, is_2d=True) for i in gt_imgs]

            # List of all the patches, ordered by image
            data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

        else:
            data = gt_imgs

    # Compute the class label of each patch based on the mean value
    if convert_to_1hot:
        labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])
    else:
        labels = data

    return labels.astype(numpy.float32)


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


def balance_dataset(data, labels):
    c0 = 0
    c1 = 0
    for i in range(len(labels)):
        if labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print('Balancing data...')
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
    print('Balanced number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    return data, labels
