NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
NUM_IMAGES = 100
TRAINING_SIZE = 80
VALIDATION_SIZE = 10
TEST_SIZE = 10
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64  # 64
NUM_EPOCHS = 100
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

"""
Context margin around the patch to be considered.
"""
PATCH_CONTEXT_SIZE = 16

EFFECTIVE_INPUT_SIZE = 2 * PATCH_CONTEXT_SIZE + IMG_PATCH_SIZE