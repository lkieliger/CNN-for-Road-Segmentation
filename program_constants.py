NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
NUM_IMAGES = 100
TRAINING_SIZE = 80
VALIDATION_SIZE = 10
TEST_SIZE = 10
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 200  # 200 for 64-wide images on 940MX LK
NUM_EPOCHS = 50
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
USE_DROPOUT = True;
DROPOUT_KEEP_RATE = 0.8;

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

"""
Context margin around the patch to be considered.
"""
PATCH_CONTEXT_SIZE = 24 #TODO: go to 20 for 3 pooling custom model

EFFECTIVE_INPUT_SIZE = 2 * PATCH_CONTEXT_SIZE + IMG_PATCH_SIZE