# This file is inspired by the segment_aerial_images.ipynb file, provided by the EPFL ML course.

import os

from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

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


def compute_scores(Z, Y):
    """
    Compute the scores between Z, the prediction and Y, the truth
    :param Z: The prediction
    :param Y: The ground truth
    :return: The accuracy score and the F1 score
    """

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
    return accuracy, f1s


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
    print('Class 0: ' + str(len(Y0)) + ' samples. Percentage: ' + str(float(len(Y0)) / float(len(Y))))
    print('Class 1: ' + str(len(Y1)) + ' samples. Percentage: ' + str(float(len(Y1)) / float(len(Y))))

    k = 0
    kf = KFold(n_splits=4)
    test_f1s = []
    test_accuracy = []
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[train_index]
        Y_train = Y[train_index]
        Y_test = Y[test_index]

        # train a logistic regression classifier
        # we create an instance of the classifier and fit the data
        logreg = linear_model.LogisticRegression(C=1e5, class_weight='balanced')
        polynomial = PolynomialFeatures(4, interaction_only=False)
        X_train_poly = polynomial.fit_transform(X_train)
        X_test_poly = polynomial.fit_transform(X_test)
        print("Training logisitic regression")
        logreg.fit(X_train_poly, Y_train)

        print("Predicting logisitic regression for train and test parts:")
        # Predict on the training set
        Z_train = logreg.predict(X_train_poly)
        Z_test = logreg.predict(X_test_poly)

        accuracy, f1s = compute_scores(Z_train, Y_train)
        print('TRAIN:\tk={}.\tF1 score: {}\tAccuracy: {}'.format(k, f1s, accuracy))
        accuracy, f1s = compute_scores(Z_test, Y_test)
        print('TEST:\tk={}.\tF1 score: {}\tAccuracy: {}'.format(k, f1s, accuracy))

        k += 1
        test_f1s.append(f1s)
        test_accuracy.append(accuracy)

    print("After {}-folds cross-validation, we have \nACCURACY: {} mean, {} std\nF1S: {} mean, {} std".
          format(k, np.mean(test_accuracy), np.std(test_accuracy), np.mean(test_f1s), np.std(test_f1s)))


if __name__ == '__main__':
    main()
