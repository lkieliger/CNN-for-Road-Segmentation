NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
NUM_IMAGES = 100
TRAINING_SIZE = 80
VALIDATION_SIZE = 20
TEST_SIZE = 0
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 128  # 200 for 64-wide images on 940MX LK
SHUFFLE_DATA = True
NUM_EPOCHS = 50
ADAM_INITIAL_LEARNING_RATE = 0.001 # more than 0.01 for custom, 0.001 for base model (maybe even less)
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
USE_DROPOUT = True
BALANCE_DATA = True
DROPOUT_KEEP_RATE = 0.8;

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

# TODO: check images output with augmented dataset and no shuffling

"""
Context margin around the patch to be considered.
"""
PATCH_CONTEXT_SIZE = 20 #TODO: go to 20 for 3 pooling custom model

EFFECTIVE_INPUT_SIZE = 2 * PATCH_CONTEXT_SIZE + IMG_PATCH_SIZE