import tensorflow as tf

from helpers.image_helpers import get_image_summary
from program_constants import *


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=SEED)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def relu(x):
    return tf.nn.relu(x)


def dropout(x):
    return tf.nn.dropout(x, DROPOUT_KEEP_RATE, seed=SEED)


class AbstractModel:
    def __init__(self, description):
        self.description = description

    def get_description(self):
        return self.description

class BaselineModel(AbstractModel):
    def __init__(self):
        AbstractModel.__init__(self, "Conv[5,5,3,32] ReLU Pool[2] Conv[5,5,32,64] ReLu Pool[2] Full[512] ReLu Full[2]")
        self._initialize_model_params()

    def _initialize_model_params(self):

        # First layer CONVOLVED 5
        self.conv1_weights = weight_variable([5, 5, NUM_CHANNELS, 32])
        self.conv1_biases = bias_variable([32])

        # Second layer CONVOLVED 5
        self.conv2_weights = weight_variable([5, 5, 32, 64])
        self.conv2_biases = bias_variable([64])

        # Third layer FULLY CONNECTED
        self.fc1_weights = weight_variable([int(EFFECTIVE_INPUT_SIZE / 4 * EFFECTIVE_INPUT_SIZE / 4 * 64), 512])
        self.fc1_biases = bias_variable([512])

        # Fourth layer FULLY CONNECTED
        self.fc2_weights = weight_variable([512, NUM_LABELS])
        self.fc2_biases = bias_variable([NUM_LABELS])

    def model_func(self):
        # We will replicate the model structure for the training subgraph, as well
        # as the evaluation subgraphs, while sharing the trainable parameters.
        def model(data, train=False):
            """The Model definition."""

            conv1 = conv2d(data, self.conv1_weights)
            relu1 = relu(conv1 + self.conv1_biases)
            pool1 = max_pool_2x2(relu1)

            if USE_DROPOUT and train:
                pool1 = dropout(pool1)

            conv2 = conv2d(pool1, self.conv2_weights)
            relu2 = relu(conv2 + self.conv2_biases)
            pool2 = max_pool_2x2(relu2)

            if USE_DROPOUT and train:
                pool2 = dropout(pool2)

            # Reshape the feature map cuboid into a 2D matrix to feed it to the
            # fully connected layers.
            pool_shape = pool2.get_shape().as_list()
            reshape = tf.reshape(
                pool2,
                shape=[-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])

            hidden = relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)

            if USE_DROPOUT and train:
                hidden = dropout(hidden)

            out = tf.matmul(hidden, self.fc2_weights) + self.fc2_biases

            return out

        return model

class CustomModel(AbstractModel):
    def __init__(self):
        AbstractModel.__init__(self, "Conv[5,5,3,32] ReLU Pool[2] Drop Conv[3,3,32,64] ReLu Pool[2] Drop"
                                     "Conv[3,3,64,128] Pool[2] Conv[3,3,64,128] Pool[2] Drop Full[128] ReLu ")
        self._initialize_model_params()

    def _initialize_model_params(self):
        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when when we call:
        # {tf.initialize_all_variables().run()}

        CONV_DEPTH1 = 32;
        CONV_DEPTH2 = 64;
        CONV_DEPTH3 = 128;
        TWO_POWER_N_POOL = 2 * 2 * 2;
        FC1_SIZE = 128;

        self.conv1_weights = weight_variable([5, 5, NUM_CHANNELS, CONV_DEPTH1])
        self.conv1_biases = bias_variable([CONV_DEPTH1])

        self.conv2_weights = weight_variable([3, 3, CONV_DEPTH1, CONV_DEPTH2])
        self.conv2_biases = bias_variable([CONV_DEPTH2])

        self.conv3_weights = weight_variable([3, 3, CONV_DEPTH2, CONV_DEPTH3])
        self.conv3_biases = bias_variable([CONV_DEPTH3])

        self.conv4_weights = weight_variable([3, 3, CONV_DEPTH3, CONV_DEPTH3])
        self.conv4_biases = bias_variable([CONV_DEPTH3])

        # Third layer FULLY CONNECTED
        self.fc1_weights = weight_variable(
            [int(EFFECTIVE_INPUT_SIZE / TWO_POWER_N_POOL * EFFECTIVE_INPUT_SIZE / TWO_POWER_N_POOL * CONV_DEPTH3),
             FC1_SIZE]
        )
        self.fc1_biases = bias_variable([FC1_SIZE])

        # Fourth layer FULLY CONNECTED
        self.fc2_weights = weight_variable([FC1_SIZE, NUM_LABELS])
        self.fc2_biases = bias_variable([NUM_LABELS])

    def model_func(self):
        # We will replicate the model structure for the training subgraph, as well
        # as the evaluation subgraphs, while sharing the trainable parameters.
        def model(data, train=False):
            """The Model definition."""
            # 2D convolution, with 'SAME' padding (i.e. the output feature map has
            # the same size as the input). Note that {strides} is a 4D array whose
            # shape matches the data layout: [image index, y, x, depth].
            conv1 = conv2d(data, self.conv1_weights)
            relu1 = relu(conv1 + self.conv1_biases)
            pool1 = max_pool_2x2(relu1)

            if USE_DROPOUT:
                pool1 = dropout(pool1);

            conv2 = conv2d(pool1, self.conv2_weights)
            relu2 = relu(conv2 + self.conv2_biases)
            pool2 = max_pool_2x2(relu2)

            if USE_DROPOUT:
                pool2 = dropout(pool2);

            conv3 = conv2d(pool2, self.conv3_weights)
            relu3 = relu(conv3 + self.conv3_biases)
            #pool3 = max_pool_2x2(relu3)

            # No dropout as in VGGNet config A

            conv4 = conv2d(relu3, self.conv4_weights)
            relu4 = relu(conv4 + self.conv4_biases)
            pool4 = max_pool_2x2(relu4)

            if USE_DROPOUT:
                pool4 = dropout(pool4);


            # Reshape the feature map cuboid into a 2D matrix to feed it to the
            # fully connected layers.
            pool_shape = pool4.get_shape().as_list()
            reshape = tf.reshape(
                pool4,
                [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])

            print(reshape)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            hidden = relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)

            if USE_DROPOUT:
                hidden = dropout(hidden)

            out = tf.matmul(hidden, self.fc2_weights) + self.fc2_biases

            return out

        return model