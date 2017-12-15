import os
import random

import matplotlib.image as mpimg
from PIL import Image
import cv2
import numpy as np

from helpers.image_helpers import img_float_to_uint8
from utils.dataset_partitioner import clean_folder

input_path_truth = '../data/training/groundtruth_original/'
input_path_images = '../data/training/images_original/'

output_path_truth = '../data/training/groundtruth/'
output_path_images = '../data/training/images/'


def generate_images(path_input, path_output, transformation, *params):
    img = cv2.imread(path_input)
    cv2.imwrite(path_output, transformation(img, *params))


def generate_rotated_training_images(angle=45, use_delta=False, override_image=True):
    num_files = 0 if override_image else len(next(os.walk(output_path_truth))[2])
    for i in range(1, 101):
        delta = random.uniform(-5, 5) if use_delta else 0
        generate_images(input_path_truth + 'satImage_' + '{num:03d}'.format(num=i) + '.png',
                        output_path_truth + 'satImage_' + '{num:03d}'.format(num=i + num_files) + '.png',
                        rotate_image, True, angle + delta)
        generate_images(input_path_images + 'satImage_' + '{num:03d}'.format(num=i) + '.png',
                        output_path_images + 'satImage_' + '{num:03d}'.format(num=i + num_files) + '.png',
                        rotate_image, False, angle + delta)


def generate_vertical_flip_training_images(override_image=True):
    num_files = 0 if override_image else len(next(os.walk(output_path_truth))[2])
    for i in range(1, 101):
        generate_images(input_path_truth + 'satImage_' + '{num:03d}'.format(num=i) + '.png',
                        output_path_truth + 'satImage_' + '{num:03d}'.format(num=i + num_files) + '.png',
                        vertical_flip, True)
        generate_images(input_path_images + 'satImage_' + '{num:03d}'.format(num=i) + '.png',
                        output_path_images + 'satImage_' + '{num:03d}'.format(num=i + num_files) + '.png',
                        vertical_flip, False)


def generate_horizontal_flip_training_images(override_image=True):
    num_files = 0 if override_image else len(next(os.walk(output_path_truth))[2])
    for i in range(1, 101):
        generate_images(input_path_truth + 'satImage_' + '{num:03d}'.format(num=i) + '.png',
                        output_path_truth + 'satImage_' + '{num:03d}'.format(num=i + num_files) + '.png',
                        horizontal_flip, True)
        generate_images(input_path_images + 'satImage_' + '{num:03d}'.format(num=i) + '.png',
                        output_path_images + 'satImage_' + '{num:03d}'.format(num=i + num_files) + '.png',
                        horizontal_flip, False)

def identity(img):
    return img

def vertical_flip(img, use_grayscale=False):
    res = cv2.flip(img, 1)
    if use_grayscale:
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res


def horizontal_flip(img, use_grayscale=False):
    res = cv2.flip(img, 0)
    if use_grayscale:
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res


def rotate_image(img, use_grayscale, angle):
    num_rows, num_cols = img.shape[:2]
    black_image = np.zeros((num_rows, num_cols, 3), np.uint8)

    vimg = cv2.flip(img, 1)
    himg = cv2.flip(img, 0)

    left = np.concatenate((black_image, np.concatenate((vimg, black_image), axis=0)), axis=0)
    middle = np.concatenate((himg, np.concatenate((img, himg), axis=0)), axis=0)
    right = np.concatenate((black_image, np.concatenate((vimg, black_image), axis=0)), axis=0)
    tot = np.concatenate((left, np.concatenate((middle, right), axis=1)), axis=1)

    num_rows_tot, num_cols_tot = tot.shape[:2]

    rotation_matrix = cv2.getRotationMatrix2D((num_cols_tot / 2, num_rows_tot / 2), angle, 1)
    img_rotation = cv2.warpAffine(tot, rotation_matrix, (num_cols_tot, num_rows_tot))

    crop_image = img_rotation[num_rows:num_rows * 2, num_cols:num_cols * 2]

    if use_grayscale:
        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

    return crop_image


def rotate_45(img):
    return rotate_image(img, False, 45)

def rotate_90(img):
    return rotate_image(img, False, 90)

def rotate_135(img):
    return rotate_image(img, False, 135)

def rotate_180(img):
    return rotate_image(img, False, 180)

def rotate_225(img):
    return rotate_image(img, False, 225)

def rotate_270(img):
    return rotate_image(img, False, 270)

def rotate_315(img):
    return rotate_image(img, False, 315)


def augment_image(im):

    functions_list = [identity,
                      vertical_flip, horizontal_flip,
                      rotate_45, rotate_90, rotate_135, rotate_180, rotate_225, rotate_270, rotate_315]

    return functions_list[np.random.randint(0,len(functions_list)-1)](im)

def augment_images(imgs, thread_pool):
    print("Augmenting epoch data")
    return np.array(thread_pool.map(augment_image, imgs))



if __name__ == '__main__':

    clean_folder(output_path_images+"*")
    clean_folder(output_path_truth+"*")
    generate_rotated_training_images(0, use_delta=False, override_image=False)
    #generate_rotated_training_images(45, use_delta=False, override_image=False)
    #generate_rotated_training_images(180, use_delta=False, override_image=False)
    #generate_rotated_training_images(270, use_delta=False, override_image=False)
    #generate_vertical_flip_training_images(override_image=False)
    #generate_horizontal_flip_training_images(override_image=False)

