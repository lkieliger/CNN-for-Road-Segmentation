import tensorflow as tf

from model import Model
from program_constants import *

class Learner:

    def __init__(self, train_size):
        self.train_size = train_size
        self._init_learner()

    def _init_learner(self):
        self.cNNModel = Model()

        self._init_nodes()
        self._init_logits()

        # Predictions for the minibatch, validation set and test set.
        self.predictions = tf.nn.softmax(self.logits)

        # We'll compute them only once in a while by calling their {eval()} method.
        #self.train_all_prediction = tf.nn.softmax(self.cNNModel.model_func()(self.train_all_data_node))

        self._init_regularizer()
        self._init_loss()
        self._init_learning_rate()
        self._init_optimizer()
        self._init_metrics()

        self._init_params_summaries()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()


    def _init_nodes(self):
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        self.data_node = tf.placeholder(
            tf.float32,
            shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
        self.labels_node = tf.placeholder(tf.float32,
                                          shape=(BATCH_SIZE, NUM_LABELS))
        #self.train_all_data_node = tf.constant(self.train_data)

    def _init_regularizer(self):
        # L2 regularization for the fully connected parameters.
        self.regularizers = (tf.nn.l2_loss(self.cNNModel.fc1_weights) + tf.nn.l2_loss(self.cNNModel.fc1_biases) +
                             tf.nn.l2_loss(self.cNNModel.fc2_weights) + tf.nn.l2_loss(self.cNNModel.fc2_biases))

    def _init_learning_rate(self):
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

    def _init_optimizer(self):
        # Use simple momentum for the optimization.
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                    0.0).minimize(self.loss,
                                                                  global_step=self.batch)

    def _init_logits(self):
        # Training computation: logits + cross-entropy loss.
        self.logits = self.cNNModel.model_func()(self.data_node, True)  # BATCH_SIZE*NUM_LABELS

    def _init_loss(self):
        # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.labels_node))

        # Add the regularization term to the loss.
        self.loss += 5e-4 * self.regularizers

        tf.summary.scalar('loss', self.loss)

    def _init_metrics(self):

        l = tf.argmax(self.labels_node, 1)
        p = tf.argmax(self.predictions, 1)

        self.true_pos, self.true_pos_op = tf.contrib.metrics.streaming_true_positives(
            labels=l,
            predictions=p
        )

        self.false_pos, self.false_pos_op = tf.contrib.metrics.streaming_false_positives(
            labels=l,
            predictions=p
        )

        self.true_neg, self.true_neg_op = tf.contrib.metrics.streaming_true_negatives(
            labels=l,
            predictions=p
        )

        self.false_neg, self.false_neg_op = tf.contrib.metrics.streaming_false_negatives(
            labels=l,
            predictions=p,
        )

    def _init_params_summaries(self):
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

    def update_feed_dictionary(self, batch_data, batch_labels):
        self.feed_dictionary = {
            self.data_node: batch_data,
            self.labels_node: batch_labels
        }

    def get_feed_dictionnary(self):
        return self.feed_dictionary

    def get_run_ops(self):
        return [self.optimizer, self.loss, self.learning_rate, self.predictions]

    def get_metric_update_ops(self):
        return [self.true_pos_op, self.false_pos_op, self.true_neg_op, self.false_neg_op]

    def get_metric_ops(self):
        return [self.true_pos, self.false_pos, self.true_neg, self.false_neg]