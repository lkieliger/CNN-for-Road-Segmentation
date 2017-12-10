import numpy as np
import os
import glob
import cv2
import shutil
from program_constants import *

PATH_PREFIX = "../"

def clean_folder(path):
    files = glob.glob(path)
    for f in files:
        os.remove(f)
        
def partition_data(train_prop, val_prop, test_prop):
    np.random.seed(SEED)

    num_images = len(next(os.walk(PATH_PREFIX + TRAIN_DATA_IMAGES_PATH))[2])
    shuffled_indices = np.random.permutation(range(num_images))

    indices_train = shuffled_indices[0:int(num_images * train_prop)]
    indices_validation = shuffled_indices[
                         int(num_images * train_prop): int(num_images * train_prop) + int(num_images * val_prop)]

    clean_folder(PATH_PREFIX + TRAIN_DATA_TRAIN_SPLIT_IMAGES_PATH + "*")
    clean_folder(PATH_PREFIX + TRAIN_DATA_VALIDATION_SPLIT_IMAGES_PATH + "*")
    clean_folder(PATH_PREFIX + TRAIN_DATA_TEST_SPLIT_IMAGES_PATH + "*")
    clean_folder(PATH_PREFIX + TRAIN_DATA_TRAIN_SPLIT_GROUNDTRUTH_PATH + "*")
    clean_folder(PATH_PREFIX + TRAIN_DATA_VALIDATION_SPLIT_GROUNDTRUTH_PATH + "*")
    clean_folder(PATH_PREFIX + TRAIN_DATA_TEST_SPLIT_GROUNDTRUTH_PATH + "*")

    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = PATH_PREFIX + TRAIN_DATA_IMAGES_PATH + imageid + ".png"
        gt_filename = PATH_PREFIX + TRAIN_DATA_GROUNDTRUTH_PATH + imageid + ".png"
        if os.path.isfile(image_filename) and os.path.isfile(gt_filename):
            img = cv2.imread(image_filename)
            gt = cv2.imread(gt_filename)
            shuffling_index = i-1
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


if __name__ == '__main__':
    partition_data(.8, .2, .0)
