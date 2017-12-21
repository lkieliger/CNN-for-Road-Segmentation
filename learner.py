import tensorflow as tf

from model import CustomModel
from program_constants import *

class Learner:

    def __init__(self):
        self._init_learner()

    def _init_learner(self):
        self.cNNModel = CustomModel()

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
        self.data_node = tf.placeholder(
            tf.float32,
            shape=(None, EFFECTIVE_INPUT_SIZE, EFFECTIVE_INPUT_SIZE, NUM_CHANNELS))
        self.labels_node = tf.placeholder(tf.float32,
                                          shape=(None, NUM_LABELS))

    def _init_regularizer(self):
        # L2 regularization for the fully connected parameters.
        self.regularizers = (tf.nn.l2_loss(self.cNNModel.fc1_weights) + tf.nn.l2_loss(self.cNNModel.fc1_biases) +
                             tf.nn.l2_loss(self.cNNModel.fc2_weights) + tf.nn.l2_loss(self.cNNModel.fc2_biases))# +
                             #tf.nn.l2_loss(self.cNNModel.fc3_weights) + tf.nn.l2_loss(self.cNNModel.fc3_biases) )

    def _init_learning_rate(self):
        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        self.batch = tf.Variable(0)

    def _init_optimizer(self):
        # Use simple momentum for the optimization.
        self.optimizer = tf.train.AdamOptimizer(ADAM_INITIAL_LEARNING_RATE).minimize(self.loss, global_step=self.batch)

    def _init_predictions(self):
        self.logits = self.cNNModel.model_func()(self.data_node, True)
        self.logits_validation = self.cNNModel.model_func()(self.data_node, False)
        self.logits_test = self.cNNModel.model_func()(self.data_node, False)

        # Predictions for the minibatch, validation set and test set.
        self.train_predictions = tf.nn.softmax(self.logits)
        self.validation_predictions = tf.nn.softmax(self.logits_validation)
        self.test_predictions = tf.nn.softmax(self.logits_test)


    def _init_loss(self):
        # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.labels_node))

        # Add the regularization term to the loss.
        if USE_L2_REGULARIZATION:
            self.loss += 5e-4 * self.regularizers


    def _init_metrics(self):
        l_train = tf.argmax(self.labels_node, 1)
        p_train = tf.argmax(self.train_predictions, 1)

        l_validation = tf.argmax(self.labels_node, 1)
        p_validation = tf.argmax(self.validation_predictions, 1)

        l_test = tf.argmax(self.labels_node, 1)
        p_test = tf.argmax(self.validation_predictions, 1)

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
        
        
        """
        TEST METRICS
        """
        self.true_test_pos, self.true_test_pos_op = tf.contrib.metrics.streaming_true_positives(
            labels=l_test,
            predictions=p_test,
            name="test"
        )

        self.false_test_pos, self.false_test_pos_op = tf.contrib.metrics.streaming_false_positives(
            labels=l_test,
            predictions=p_test,
            name="test"
        )

        self.true_test_neg, self.true_test_neg_op = tf.contrib.metrics.streaming_true_negatives(
            labels=l_test,
            predictions=p_test,
            name="test"
        )

        self.false_test_neg, self.false_test_neg_op = tf.contrib.metrics.streaming_false_negatives(
            labels=l_test,
            predictions=p_test,
            name="test"
        )

    def model(self):
        return self.cNNModel

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
    
    def get_test_ops(self):
        return [self.test_predictions]

    def get_test_metric_ops(self):
        return [self.true_test_pos, self.false_test_pos, 
                self.true_test_neg, self.false_test_neg]

    def get_test_metric_update_ops(self):
        return [self.true_test_pos_op, self.false_test_pos_op, 
                self.true_test_neg_op, self.false_test_neg_op]