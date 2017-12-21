import glob
import os

import cv2
import numpy as np

from helpers.data_helpers import extract_all_data, extract_all_labels
from program_constants import *

PATH_PREFIX = "../"
FILENAME = "nparray.npy"


def clean_folder(path):
    """
    Remove all files in the path, and create folders if they are not
    """
    files = glob.glob(path)
    for f in files:
        os.remove(f)
    if not os.path.isdir(path[:len(path) - 2]):
        os.mkdir(path[:len(path) - 2])



def clean_all_folders():
    """
    Remove every files in the train, test, validation folders of ground truth and satellite images.
    """
    clean_folder(PATH_PREFIX + TRAIN_DATA_TRAIN_SPLIT_IMAGES_PATH + "*")
    clean_folder(PATH_PREFIX + TRAIN_DATA_VALIDATION_SPLIT_IMAGES_PATH + "*")
    clean_folder(PATH_PREFIX + TRAIN_DATA_TEST_SPLIT_IMAGES_PATH + "*")
    clean_folder(PATH_PREFIX + TRAIN_DATA_TRAIN_SPLIT_GROUNDTRUTH_PATH + "*")
    clean_folder(PATH_PREFIX + TRAIN_DATA_VALIDATION_SPLIT_GROUNDTRUTH_PATH + "*")
    clean_folder(PATH_PREFIX + TRAIN_DATA_TEST_SPLIT_GROUNDTRUTH_PATH + "*")


def save_patches(patches, filename):
    for i, patch in enumerate(patches):
        patch.save(filename + "patch" + str(i) + ".png")


def partition_data(train_prop, val_prop, test_prop, make_patches=False):
    """
    Split the images into train test validation and save them as images.
    :param train_prop: The proportion of train
    :param val_prop:  The proportion of validation
    :param test_prop:  The proportion of test
    """
    np.random.seed(SEED)

    num_images = len(next(os.walk(PATH_PREFIX + TRAIN_DATA_IMAGES_PATH))[2])
    shuffled_indices = np.random.permutation(range(num_images))

    indices_train = shuffled_indices[0:int(num_images * train_prop)]
    indices_validation = shuffled_indices[
                         int(num_images * train_prop): int(num_images * train_prop) + int(num_images * val_prop)]

    clean_all_folders()

    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = PATH_PREFIX + TRAIN_DATA_IMAGES_PATH + imageid + ".png"
        gt_filename = PATH_PREFIX + TRAIN_DATA_GROUNDTRUTH_PATH + imageid + ".png"
        if os.path.isfile(image_filename) and os.path.isfile(gt_filename):
            img = cv2.imread(image_filename)
            gt = cv2.imread(gt_filename)
            shuffling_index = i - 1
            if shuffling_index in indices_train:
                cv2.imwrite(PATH_PREFIX + TRAIN_DATA_TRAIN_SPLIT_IMAGES_PATH + imageid + ".png", img)
                cv2.imwrite(PATH_PREFIX + TRAIN_DATA_TRAIN_SPLIT_GROUNDTRUTH_PATH + imageid + ".png", gt)
            elif shuffling_index in indices_validation:
                cv2.imwrite(PATH_PREFIX + TRAIN_DATA_VALIDATION_SPLIT_IMAGES_PATH + imageid + ".png", img)
                cv2.imwrite(PATH_PREFIX + TRAIN_DATA_VALIDATION_SPLIT_GROUNDTRUTH_PATH + imageid + ".png", gt)
            else:
                cv2.imwrite(PATH_PREFIX + TRAIN_DATA_TEST_SPLIT_IMAGES_PATH + imageid + ".png", img)
                cv2.imwrite(PATH_PREFIX + TRAIN_DATA_TEST_SPLIT_GROUNDTRUTH_PATH + imageid + ".png", gt)

        else:
            print('File ' + image_filename + ' does not exist')


def partition_patches(train_prop, val_prop, test_prop):
    """
    Split the images into train test validation and save them as numpy array
    :param train_prop: The proportion of train
    :param val_prop:  The proportion of validation
    :param test_prop:  The proportion of test
    """
    np.random.seed(SEED)

    clean_all_folders()

    data = extract_all_data(PATH_PREFIX + TRAIN_DATA_IMAGES_PATH)
    labels = extract_all_labels(PATH_PREFIX + TRAIN_DATA_GROUNDTRUTH_PATH, num_images=-1)

    num_patches = data.shape[0]

    shuffled_indices = np.random.permutation(range(num_patches))

    indices_train = shuffled_indices[0:int(num_patches * train_prop)]
    indices_validation = shuffled_indices[int(num_patches * train_prop): int(num_patches * train_prop) + int(num_patches * val_prop)]
    indices_test = shuffled_indices[int(num_patches * train_prop) + int(num_patches * val_prop):]

    np.save(PATH_PREFIX + TRAIN_DATA_TRAIN_SPLIT_IMAGES_PATH + FILENAME, data[indices_train])
    np.save(PATH_PREFIX + TRAIN_DATA_TRAIN_SPLIT_GROUNDTRUTH_PATH + FILENAME, labels[indices_train])

    np.save(PATH_PREFIX + TRAIN_DATA_VALIDATION_SPLIT_IMAGES_PATH + FILENAME, data[indices_validation])
    np.save(PATH_PREFIX + TRAIN_DATA_VALIDATION_SPLIT_GROUNDTRUTH_PATH + FILENAME, labels[indices_validation])

    np.save(PATH_PREFIX + TRAIN_DATA_TEST_SPLIT_IMAGES_PATH + FILENAME, data[indices_test])
    np.save(PATH_PREFIX + TRAIN_DATA_TEST_SPLIT_GROUNDTRUTH_PATH + FILENAME, labels[indices_test])


def read_partitions(prefix=''):
    """
    Read the partitions of the data (test, validation and train).
    :param prefix: The prefix of the path
    :return: The train, test, validation images of ground truth and satellite.
    """
    d_tr = np.load(prefix + TRAIN_DATA_TRAIN_SPLIT_IMAGES_PATH + FILENAME)
    l_tr = np.load(prefix + TRAIN_DATA_TRAIN_SPLIT_GROUNDTRUTH_PATH + FILENAME)

    d_val = np.load(prefix + TRAIN_DATA_VALIDATION_SPLIT_IMAGES_PATH + FILENAME)
    l_val = np.load(prefix + TRAIN_DATA_VALIDATION_SPLIT_GROUNDTRUTH_PATH + FILENAME)

    d_te = np.load(prefix + TRAIN_DATA_TEST_SPLIT_IMAGES_PATH + FILENAME)
    l_te = np.load(prefix + TRAIN_DATA_TEST_SPLIT_GROUNDTRUTH_PATH + FILENAME)

    return d_tr, d_val, d_te, l_tr, l_val, l_te


if __name__ == '__main__':
    partition_patches(TRAINING_PROP, VALIDATION_PROP, TEST_PROP)
    # read_partitions(PATH_PREFIX)
