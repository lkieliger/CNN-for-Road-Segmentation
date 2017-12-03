import tensorflow as tf

from model import Model
from program_constants import *

class Learner:

    def __init__(self, train_data, train_size):
        self.train_data = train_data
        self.train_size = train_size

        self.cNNModel = Model()

        self.init_nodes()
        self.init_logits()
        self.init_regularizer()
        self.init_loss()
        self.init_learning_rate()
        self.init_optimizer()

        self.init_params_summaries()

        # Predictions for the minibatch, validation set and test set.
        self.train_prediction = tf.nn.softmax(self.logits)

        # We'll compute them only once in a while by calling their {eval()} method.
        self.train_all_prediction = tf.nn.softmax(self.cNNModel.model_func()(self.train_all_data_node))

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    def init_nodes(self):
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        self.train_data_node = tf.placeholder(
            tf.float32,
            shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
        self.train_labels_node = tf.placeholder(tf.float32,
                                                shape=(BATCH_SIZE, NUM_LABELS))
        self.train_all_data_node = tf.constant(self.train_data)

    def init_regularizer(self):
        # L2 regularization for the fully connected parameters.
        self.regularizers = (tf.nn.l2_loss(self.cNNModel.fc1_weights) + tf.nn.l2_loss(self.cNNModel.fc1_biases) +
                             tf.nn.l2_loss(self.cNNModel.fc2_weights) + tf.nn.l2_loss(self.cNNModel.fc2_biases))

    def init_learning_rate(self):
        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        self.batch = tf.Variable(0)

        # Decay once per epoch, using an exponential schedule starting at 0.01.
        self.learning_rate = tf.train.exponential_decay(
            0.01,  # Base learning rate.
            self.batch * BATCH_SIZE,  # Current index into the dataset.
            self.train_size,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
        tf.summary.scalar('learning_rate', self.learning_rate)

    def init_optimizer(self):
        # Use simple momentum for the optimization.
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                    0.0).minimize(self.loss,
                                                                  global_step=self.batch)

    def init_logits(self):
        # Training computation: logits + cross-entropy loss.
        self.logits = self.cNNModel.model_func()(self.train_data_node, True)  # BATCH_SIZE*NUM_LABELS

    def init_loss(self):
        # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.train_labels_node))

        # Add the regularization term to the loss.
        self.loss += 5e-4 * self.regularizers

        tf.summary.scalar('loss', self.loss)

    def init_params_summaries(self):
        all_params_node = [self.cNNModel.conv1_weights, self.cNNModel.conv1_biases,
                           self.cNNModel.conv2_weights, self.cNNModel.conv2_biases,
                           self.cNNModel.fc1_weights, self.cNNModel.fc1_biases,
                           self.cNNModel.fc2_weights, self.cNNModel.fc2_biases]

        all_params_names = ['conv1_weights', 'conv1_biases',
                            'conv2_weights', 'conv2_biases',
                            'fc1_weights', 'fc1_biases',
                            'fc2_weights', 'fc2_biases']

        all_grads_node = tf.gradients(self.loss, all_params_node)
        all_grad_norms_node = []

        for i in range(0, len(all_grads_node)):
            norm_grad_i = tf.global_norm([all_grads_node[i]])
            all_grad_norms_node.append(norm_grad_i)
            tf.summary.scalar(all_params_names[i], norm_grad_i)