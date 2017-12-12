"""
Path constants
"""
MODEL_PATH = "models/"
TEST_DATA_PATH = "data/test_set_images/"
TRAIN_DATA_PATH = "data/training/"
TRAIN_DATA_IMAGES_PATH = TRAIN_DATA_PATH + "images/"
TRAIN_DATA_GROUNDTRUTH_PATH = TRAIN_DATA_PATH + "groundtruth/"
TRAIN_DATA_TRAIN_SPLIT_IMAGES_PATH = TRAIN_DATA_PATH + "images_train_split/"
TRAIN_DATA_VALIDATION_SPLIT_IMAGES_PATH =TRAIN_DATA_PATH + "images_validation_split/"
TRAIN_DATA_TEST_SPLIT_IMAGES_PATH =TRAIN_DATA_PATH + "images_test_split/"
TRAIN_DATA_TRAIN_SPLIT_GROUNDTRUTH_PATH = TRAIN_DATA_PATH + "groundtruth_train_split/"
TRAIN_DATA_VALIDATION_SPLIT_GROUNDTRUTH_PATH =TRAIN_DATA_PATH + "groundtruth_validation_split/"
TRAIN_DATA_TEST_SPLIT_GROUNDTRUTH_PATH =TRAIN_DATA_PATH + "groundtruth_test_split/"


# 23 epochs

NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_PROP = 1.0
VALIDATION_PROP = 0.0
TEST_PROP = 0.0
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 128  # 200 for 64-wide images on 940MX LK
SHUFFLE_DATA = False
NUM_EPOCHS = 150
ADAM_INITIAL_LEARNING_RATE = 0.001 # more than 0.01 for custom, 0.001 for base model (maybe even less)
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
USE_DROPOUT = True
USE_L2_REGULARIZATION = True
USE_LEAKY_RELU = False
BALANCE_TRAIN_DATA = False
DROPOUT_KEEP_RATE = 0.5
IMG_PATCH_SIZE = 16

# TODO: check images output with augmented dataset and no shuffling

"""
Context margin around the patch to be considered.
"""
PATCH_CONTEXT_SIZE = 16 #TODO: go to 20 for 3 pooling custom model

EFFECTIVE_INPUT_SIZE = 2 * PATCH_CONTEXT_SIZE + IMG_PATCH_SIZE