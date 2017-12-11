import sys
import numpy as np
import datetime

from config_logger import ConfigLogger
from helpers.plots import *
from helpers.prediction_helpers import *
from helpers.data_helpers import *
from learner import Learner
from metrics import *
from model import *
from program_constants import *
from utils.dataset_partitioner import read_partitions

now = datetime.datetime.now()
FLAGS = tf.app.flags.FLAGS

def output_training_set_results(session, learner):
    print("Running prediction on training set")
    prediction_training_dir = "predictions_training/"

    if not os.path.isdir(prediction_training_dir):
        os.mkdir(prediction_training_dir)

    for j, filename in enumerate(os.listdir(TRAIN_DATA_IMAGES_PATH)):
        i = j+1
        #pimg = get_prediction_with_groundtruth(train_data_filename, i, learner.cNNModel, session)
        #Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
        oimg = get_prediction_with_overlay(TRAIN_DATA_IMAGES_PATH, i, learner.cNNModel, session)
        oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")



def assess_model(session, learner, data, labels):

    data_indices = range(data.shape[0])

    # Feed test data to the model batch by batch and compute running statistics
    for i in range(0, data.shape[0], BATCH_SIZE):
        batch_indices = data_indices[i: i + BATCH_SIZE]
        batch_data = data[batch_indices]
        batch_labels = labels[batch_indices]

        learner.update_feed_dictionary(batch_data, batch_labels)

        # Run the graph and fetch some of the nodes.
        predictions, _, _, _, _ = session.run(
            learner.get_test_ops() + learner.get_test_metric_update_ops(),
            feed_dict=learner.get_feed_dictionnary())

    # Evaluate the final test statistics
    tp, fp, tn, fn = session.run(learner.get_test_metric_ops())
    acc = accuracy(tp, fp, tn, fn)
    pre = precision(tp, fp)
    rec = recall(tp, fn)
    f1s = f1_score(tp, fp, fn)
    print("\t [ TEST REPORT ]")
    print("\t F1: {:.2%}, Accuracy: {:.2%}, Precision: {:.2%}, Recall: {:.2%}".format(f1s, acc, pre, rec))
    print("\t TP: {}, TN: {}, FP: {}, FN: {} \n".format(tp, tn, fp, fn))

    return f1s, acc, pre, rec


def main(argv=None):  # pylint: disable=unused-argument

    data_dir = 'data/training/'

    data_train, data_validation, data_test, labels_train, labels_validation, labels_test = read_partitions()

    if BALANCE_TRAIN_DATA:
        data_train, labels_train = balance_dataset(data_train, labels_train)

    print("Training data shape: {}".format(data_train.shape))
    print("Validation data shape: {}".format(data_validation.shape))
    #print("Test data shape: {}".format(data_test.shape))

    learner = Learner(data_train.shape[0])

    accuracy_data_training = []
    accuracy_data_validation = []
    logger = ConfigLogger()
    logger.describe_model(learner.cNNModel.get_description())
    weigths_1 = []

    # Create a local session to run this computation.
    with tf.Session() as tensorflow_session:

        if RESTORE_MODEL:
            # Restore variables from disk.
            learner.saver.restore(tensorflow_session, "restore/model.ckpt")
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            init = tf.global_variables_initializer()
            init_loc = tf.local_variables_initializer()
            tensorflow_session.run(init)
            tensorflow_session.run(init_loc)

            print('Initialized!')
            print('Total number of iterations = ' + str(int(NUM_EPOCHS * data_train.shape[0] / BATCH_SIZE)))

            training_indices = range(data_train.shape[0])
            validation_indices = range(data_validation.shape[0])

            for iepoch in range(NUM_EPOCHS):
                # Reset local variables, needed for metrics
                tensorflow_session.run(init_loc)
                print("")
                print("=============================================================")
                print(" RUNNING EPOCH {}                                          ".format(iepoch))
                print("=============================================================")
                # Permute training indices
                perm_indices_train = numpy.random.permutation(training_indices)
                perm_indices_validation = numpy.random.permutation(validation_indices)

                # Train on whole dataset, batch by batch
                for step in range(int(data_train.shape[0] / BATCH_SIZE)):

                    offset = (step * BATCH_SIZE) % (data_train.shape[0] - BATCH_SIZE)
                    batch_indices = perm_indices_train[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = data_train[batch_indices, :, :, :]
                    batch_labels = labels_train[batch_indices]
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    learner.update_feed_dictionary(batch_data, batch_labels)

                    # Run the graph and fetch some of the nodes.
                    _, l, predictions, _, _, _, _= tensorflow_session.run(
                        learner.get_train_ops() + learner.get_train_metric_update_ops(),
                        feed_dict=learner.get_feed_dictionnary())

                    #print_predictions(predictions, batch_labels)

                # Assess performance by running on validation dataset, batch by batch
                for step in range(int(data_validation.shape[0] / BATCH_SIZE)):
                    offset = (step * BATCH_SIZE) % (data_validation.shape[0] - BATCH_SIZE)
                    batch_indices = perm_indices_validation[offset:(offset + BATCH_SIZE)]

                    batch_data = data_validation[batch_indices, :, :, :]
                    batch_labels = labels_validation[batch_indices]

                    learner.update_feed_dictionary(batch_data, batch_labels)

                    # Run the graph and fetch some of the nodes.
                    predictions, _, _, _, _ = tensorflow_session.run(
                        learner.get_validation_ops() + learner.get_validation_metric_update_ops(),
                        feed_dict=learner.get_feed_dictionnary())


                """
                TRAINING REPORT
                """
                tp, fp, tn, fn = tensorflow_session.run(learner.get_train_metric_ops())
                acc = accuracy(tp, fp, tn, fn)
                pre = precision(tp, fp)
                rec = recall(tp, fn)
                f1s = f1_score(tp, fp, fn)
                print("\t [ TRAINING REPORT ]")
                print("\t F1: {:.2%}, Accuracy: {:.2%}, Precision: {:.2%}, Recall: {:.2%}".format(f1s, acc, pre, rec))
                print("\t TP: {}, TN: {}, FP: {}, FN: {} \n".format(tp, tn, fp, fn))

                logger.set_train_score(acc, pre, rec, f1s)

                accuracy_data_training.append(acc)

                """
                VALIDATION REPORT
                """
                tp, fp, tn, fn = tensorflow_session.run(learner.get_validation_metric_ops())
                acc = accuracy(tp, fp, tn, fn)
                pre = precision(tp, fp)
                rec = recall(tp, fn)
                f1s = f1_score(tp, fp, fn)
                print("\t [ VALIDATION REPORT ]")
                print("\t F1: {:.2%}, Accuracy: {:.2%}, Precision: {:.2%}, Recall: {:.2%}".format(f1s, acc, pre, rec))
                print("\t TP: {}, TN: {}, FP: {}, FN: {} \n".format(tp, tn, fp, fn))

                logger.set_validation_score(acc, pre, rec, f1s)

                accuracy_data_validation.append(acc)

                # Save the variables to disk.
                save_path = learner.saver.save(tensorflow_session, "models/model.ckpt")
                print("\t Model saved in file: %s" % save_path)

            # Compute stats on test set
            f1s, acc, pre, rec = assess_model(tensorflow_session, learner, data_test, labels_test)
            logger.set_test_score(acc, pre, rec, f1s)

            logger.save()
            plot_accuracy([accuracy_data_training, accuracy_data_validation], acc, logger.get_timestamp())

            weigths_1 = tensorflow_session.run(learner.cNNModel.conv1_weights)
            plot_conv_weights(weigths_1, logger.get_timestamp())

        print("=============================================================")
        print("=============================================================")
        print("")

        output_training_set_results(tensorflow_session, learner)




if __name__ == '__main__':
    tf.app.run()
