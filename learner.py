import tensorflow as tf

from model import BaselineModel, CustomModel
from program_constants import *

class Learner:

    def __init__(self, train_size):
        self.train_size = train_size
        self._init_learner()

    def _init_learner(self):
        self.cNNModel = BaselineModel()

        self._init_nodes()
        self._init_predictions()

        # We'll compute them only once in a while by calling their {eval()} method.
        #self.train_all_prediction = tf.nn.softmax(self.cNNModel.model_func()(self.train_all_data_node))

        self._init_regularizer()
        self._init_loss()
        self._init_learning_rate()
        self._init_optimizer()
        self._init_metrics()

        #self._init_params_summaries()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()


    def _init_nodes(self):
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        self.data_node = tf.placeholder(
            tf.float32,
            shape=(BATCH_SIZE, EFFECTIVE_INPUT_SIZE, EFFECTIVE_INPUT_SIZE, NUM_CHANNELS))
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

    def _init_optimizer(self):
        # Use simple momentum for the optimization.
        self.optimizer = tf.train.AdamOptimizer(ADAM_INITIAL_LEARNING_RATE).minimize(self.loss, global_step=self.batch)

    def _init_predictions(self):
        self.logits = self.cNNModel.model_func()(self.data_node, True)  # BATCH_SIZE*NUM_LABELS
        self.logits_validation = self.cNNModel.model_func()(self.data_node, False)  # BATCH_SIZE*NUM_LABELS

        # Predictions for the minibatch, validation set and test set.
        self.train_predictions = tf.nn.softmax(self.logits)
        self.validation_predictions = tf.nn.softmax(self.logits_validation)


    def _init_loss(self):
        # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.labels_node))

        # Add the regularization term to the loss.
        self.loss += 5e-4 * self.regularizers

        tf.summary.scalar('loss', self.loss)

    def _init_metrics(self):
        l_train = tf.argmax(self.labels_node, 1)
        p_train = tf.argmax(self.train_predictions, 1)

        l_validation = tf.argmax(self.labels_node, 1)
        p_validation = tf.argmax(self.validation_predictions, 1)

        """
        TRAINING METRICS
        """
        self.true_train_pos, self.true_train_pos_op = tf.contrib.metrics.streaming_true_positives(
            labels=l_train,
            predictions=p_train,
            name="train"
        )

        self.false_train_pos, self.false_train_pos_op = tf.contrib.metrics.streaming_false_positives(
            labels=l_train,
            predictions=p_train,
            name="train"
        )

        self.true_train_neg, self.true_train_neg_op = tf.contrib.metrics.streaming_true_negatives(
            labels=l_train,
            predictions=p_train,
            name="train"
        )

        self.false_train_neg, self.false_train_neg_op = tf.contrib.metrics.streaming_false_negatives(
            labels=l_train,
            predictions=p_train,
            name="train"
        )

        """
        VALIDATION METRICS
        """
        self.true_validation_pos, self.true_validation_pos_op = tf.contrib.metrics.streaming_true_positives(
            labels=l_validation,
            predictions=p_validation,
            name="validation"
        )

        self.false_validation_pos, self.false_validation_pos_op = tf.contrib.metrics.streaming_false_positives(
            labels=l_validation,
            predictions=p_validation,
            name="validation"
        )

        self.true_validation_neg, self.true_validation_neg_op = tf.contrib.metrics.streaming_true_negatives(
            labels=l_validation,
            predictions=p_validation,
            name="validation"
        )

        self.false_validation_neg, self.false_validation_neg_op = tf.contrib.metrics.streaming_false_negatives(
            labels=l_validation,
            predictions=p_validation,
            name="validation"
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

    def get_train_ops(self):
        return [self.optimizer, self.loss, self.train_predictions]

    def get_train_metric_ops(self):
        return [self.true_train_pos, self.false_train_pos, 
                self.true_train_neg, self.false_train_neg]

    def get_train_metric_update_ops(self):
        return [self.true_train_pos_op, self.false_train_pos_op, 
                self.true_train_neg_op, self.false_train_neg_op]

    def get_validation_ops(self):
        return [self.validation_predictions]

    def get_validation_metric_ops(self):
        return [self.true_validation_pos, self.false_validation_pos, 
                self.true_validation_neg, self.false_validation_neg]

    def get_validation_metric_update_ops(self):
        return [self.true_validation_pos_op, self.false_validation_pos_op, 
                self.true_validation_neg_op, self.false_validation_neg_op]