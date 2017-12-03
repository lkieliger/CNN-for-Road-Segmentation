import tensorflow as tf

from helpers.image_helpers import get_image_summary
from program_constants import *


class Model:

    def __init__(self):
        self._initialize_model_params()


    def _initialize_model_params(self):
        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when when we call:
        # {tf.initialize_all_variables().run()}

        # First layer CONVOLVED 5
        self.conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                                stddev=0.1,
                                seed=SEED))
        self.conv1_biases = tf.Variable(tf.zeros([32]))

        # Second layer CONVOLVED 5
        self.conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 64],
                                stddev=0.1,
                                seed=SEED))
        self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

        # Third layer FULLY CONNECTED
        self.fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal([int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
                                stddev=0.1,
                                seed=SEED))
        self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

        # Fourth layer FULLY CONNECTED
        self.fc2_weights = tf.Variable(
            tf.truncated_normal([512, NUM_LABELS],
                                stddev=0.1,
                                seed=SEED))
        self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))


    def model_func(self):

        # We will replicate the model structure for the training subgraph, as well
        # as the evaluation subgraphs, while sharing the trainable parameters.
        def model(data, train=False):
            """The Model definition."""
            # 2D convolution, with 'SAME' padding (i.e. the output feature map has
            # the same size as the input). Note that {strides} is a 4D array whose
            # shape matches the data layout: [image index, y, x, depth].
            conv = tf.nn.conv2d(data,
                                self.conv1_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')
            # Bias and rectified linear non-linearity.
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
            # Max pooling. The kernel size spec {ksize} also follows the layout of
            # the data. Here we have a pooling window of 2, and a stride of 2.
            pool = tf.nn.max_pool(relu,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')

            conv2 = tf.nn.conv2d(pool,
                                 self.conv2_weights,
                                 strides=[1, 1, 1, 1],
                                 padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.conv2_biases))
            pool2 = tf.nn.max_pool(relu2,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

            # Uncomment these lines to check the size of each layer
            # print 'data ' + str(data.get_shape())
            # print 'conv ' + str(conv.get_shape())
            # print 'relu ' + str(relu.get_shape())
            # print 'pool ' + str(pool.get_shape())
            # print 'pool2 ' + str(pool2.get_shape())


            # Reshape the feature map cuboid into a 2D matrix to feed it to the
            # fully connected layers.
            pool_shape = pool2.get_shape().as_list()
            reshape = tf.reshape(
                pool2,
                [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
            # Add a 50% dropout during training only. Dropout also scales
            # activations such that no rescaling is needed at evaluation time.
            # if train:
            #    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
            out = tf.matmul(hidden, self.fc2_weights) + self.fc2_biases

            if train == True:
                summary_id = '_0'
                s_data = get_image_summary(data)
                filter_summary0 = tf.summary.image('summary_data' + summary_id, s_data)
                s_conv = get_image_summary(conv)
                filter_summary2 = tf.summary.image('summary_conv' + summary_id, s_conv)
                s_pool = get_image_summary(pool)
                filter_summary3 = tf.summary.image('summary_pool' + summary_id, s_pool)
                s_conv2 = get_image_summary(conv2)
                filter_summary4 = tf.summary.image('summary_conv2' + summary_id, s_conv2)
                s_pool2 = get_image_summary(pool2)
                filter_summary5 = tf.summary.image('summary_pool2' + summary_id, s_pool2)

            return out

        return model