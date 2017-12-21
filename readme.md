# Road segmentation - ML Project 2

### Model used

In order to determine which parts of an image belong to the `road` class vs the `background` class, a 4  layer Convolutional Neural Network is used.

The square input images (of width *400px*) are split in *16 px* width square patches, together with the patch context (that is, *16 px* of context next to the analyzed patch, and a total considered patch of size *48x48px*). 

The various settings can be selected in the `program_constants.py` file, and their signification is explained below.

The model is defined in the `model.py` module, and can be trained using the `tf_aerial_images.py` module The library responsible for doing the computations is [Tensorflow](https://www.tensorflow.org/).

Before anything, the data (`training` and `test_set_images` folders) must be placed in the data folder at the root.


### Model settings

These parameters can be set to convenience in the `program_constants.py` file (only the one which require an explanation are stated in the table):

| Variable name              | Effect                                   |
| -------------------------- | ---------------------------------------- |
| NUM_CHANNELS               | Defines the number of channel of the input images (e.g. 3 for RGB) |
| PIXEL_DEPTH                | Number of colour of a pixel              |
| TEST_PROP                  | Proportion of the dataset to be used to test the model |
| SEED                       | Definition of the seed use for the random number generators |
| BATCH_SIZE                 | Number of patches to be processed at once by tensorflow |
| NUM_EPOCHS                 | Defines how many times the whole dataset will be processed by the learning algorithm |
| ADAM_INITIAL_LEARNING_RATE | Defines the learning rate for the Adam optimizer |
| USE_DROPOUT                | If enabled, a random proportion of the neurons are not updated during an iteration of the training of the model |
| USE_L2_REGULARIZATION      | Defines whether to use L2 regularization or not |
| USE_LEAKY_RELU             | Defines whether to use a leaky rectified linear unit activation function or not |
| DROPOUT_KEEP_RATE          | Proportion of the neurons that are not "dropped out". Used in conjonction with the ÙSE_DROPOUT`setting |
| IMG_PATCH_SIZE             | Width of an image part.                  |
| PATCH_CONTEXT_SIZE         | Width of the neighboring pixels to be considered together with the image patch. |
| RESTORE_MODEL_NAME         | Name of the model to restore             |

### Training the model

As the computations involve millions of parameters update, we used Amazon AWS remote instances, which provided us access to NVidia Tesla K80 GPU, so as to speedup the model training.

Before training, the data must be serialized and partitioned between the train, test and validation set.  To do this, run the following commands at the root of the project:

	python -m utils.dataset_partitioner
Then, to train the model, use: `python tf_aerial_images.py`

Once a model is trained, use the `python run.py` command that will output the predictions using the model specified with the `RESTORE_MODEL_NAME` setting. The default restored model, `cnnb-full-175_model` is the model that was trained for 175 epochs on the full dataset.

After having run the `run.py` file, the submission for Kaggle is created in the `submissions/`folder.

#### Module dependencies

- Tensorflow (version 1.14)
- Numpy (version 1.13)
- OpenCV (version 3.3.0.10)
- Pandas (version 0.20.1)