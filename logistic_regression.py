# This file is inspired by the segment_aerial_images.ipynb file, provided by the EPFL ML course.

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import os, sys

# Loaded a set of images
from helpers.image_helpers import *


def load_images():
    """
    Load images and extract patches of size 16x16 of each images from the training.
    :return: Return an array containing patches of size 16x16 from truth and satellite images.
    """
    root_dir = "data/training/"
    image_dir = root_dir + "images/"
    files = os.listdir(image_dir)
    n = min(100, len(files))  # Load maximum 100 images
    print("Loading " + str(n) + " images")
    imgs = [mpimg.imread(image_dir + files[i]) for i in range(n)]

    gt_dir = root_dir + "groundtruth/"
    print("Loading " + str(n) + " images")
    gt_imgs = [mpimg.imread(gt_dir + files[i]) for i in range(n)]

    n = 100  # Only use 10 images for training

    # Extract patches from input images
    patch_size = 16  # each patch is 16*16 pixels

    img_patches = [img_crop(imgs[i], patch_size, patch_size, is_2d=True, patch_context_override=0)
                   for i in range(n)]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size, is_2d=True, patch_context_override=0)
                  for i in range(n)]

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    return img_patches, gt_patches


def extract_features(img):
    """ Extract 6-dimensional features consisting of average RGB color as well as variance
    """
    feat_m = np.mean(img, axis=(0, 1))
    feat_v = np.var(img, axis=(0, 1))
    feat = np.append(feat_m, feat_v)
    return feat


def extract_features_2d(img):
    """Extract 2-dimensional features consisting of average gray color as well as variance
    """
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat


def value_to_class(v):
    """ Compute features for each image patch
    """
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch

    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def main():
    img_patches, gt_patches = load_images()

    X = np.asarray([extract_features(img_patches[i]) for i in range(len(img_patches))])
    Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])

    # Print feature statistics
    print('Computed ' + str(X.shape[0]) + ' features')
    print('Feature dimension = ' + str(X.shape[1]))
    print('Number of classes = ' + str(np.max(Y)))

    Y0 = [i for i, j in enumerate(Y) if j == 0]
    Y1 = [i for i, j in enumerate(Y) if j == 1]
    print('Class 0: ' + str(len(Y0)) + ' samples')
    print('Class 1: ' + str(len(Y1)) + ' samples')

    # Display a patch that belongs to the foreground class
    plt.imshow(gt_patches[Y1[6]], cmap='Greys_r')

    # train a logistic regression classifier

    # we create an instance of the classifier and fit the data
    logreg = linear_model.LogisticRegression(C=1e5, class_weight='balanced')
    polynomial = PolynomialFeatures(4, interaction_only=False)
    X_poly = polynomial.fit_transform(X)
    print("Training logisitic regression")
    logreg.fit(X_poly, Y)

    print("Predicting logisitic regression")
    # Predict on the training set
    Z = logreg.predict(X_poly)

    # Get non-zeros in prediction and grountruth arrays
    Z_true = np.nonzero(Z)[0]
    Z_false = np.nonzero(Z - 1)[0]
    Y_true = np.nonzero(Y)[0]
    Y_false = np.nonzero(Y - 1)[0]

    true_positive = len(list(set(Y_true) & set(Z_true)))
    false_positive = len(list(set(Y_false) & set(Z_true)))
    true_negative = len(list(set(Y_false) & set(Z_false)))
    false_negative = len(list(set(Y_true) & set(Z_false)))
    recall = true_positive / float(len(Y_true))
    precision = true_positive / float(true_positive + false_positive)
    accuracy = float(true_positive + true_negative) / \
               float(true_negative + true_positive + false_positive + false_negative)

    f1s = 2 * (precision * recall) / (precision + recall)
    print('Precision = ' + str(precision) + "\tRecall = " + str(recall))
    print('F1 score: ' + str(f1s) + "\tAccuracy: " + str(accuracy))


if __name__ == '__main__':
    main()
