import sys

from config_logger import ConfigLogger
from helpers.image_helpers import *
from helpers.prediction_helpers import *
from helpers.data_helpers import *
from learner import Learner
from metrics import *
from model import *
from plots import *
from program_constants import *

tf.app.flags.DEFINE_string('train_dir', '/tmp/mnist',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS

def output_training_set_results(session, learner, train_data_filename):
    print("Running prediction on training set")
    prediction_training_dir = "predictions_training/"
    if not os.path.isdir(prediction_training_dir):
        os.mkdir(prediction_training_dir)
    for i in range(1, TRAINING_SIZE + 1):
        pimg = get_prediction_with_groundtruth(train_data_filename, i, learner.cNNModel, session)
        Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
        oimg = get_prediction_with_overlay(train_data_filename, i, learner.cNNModel, session)
        oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")

def main(argv=None):  # pylint: disable=unused-argument

    data_dir = 'data/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'

    # Extract it into numpy arrays.
    data = extract_data(train_data_filename, NUM_IMAGES)
    labels = extract_labels(train_labels_filename, NUM_IMAGES)

    num_epochs = NUM_EPOCHS

    c0 = 0
    c1 = 0
    for i in range(len(labels)):
        if labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    # TODO: shuffle before balancing ?
    print('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print(len(new_indices))
    print(data.shape)
    data = data[new_indices, :, :, :]
    labels = labels[new_indices]

    data_size = labels.shape[0]
    print(data_size)

    c0 = 0
    c1 = 0
    for i in range(len(labels)):
        if labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))


    # Split data
    data_train, data_validation, data_test, labels_train, labels_validation, labels_test = split_data(data, labels)

    print("Training data shape: {}".format(data_train.shape))
    print("Validation data shape: {}".format(data_validation.shape))
    print("Test data shape: {}".format(data_test.shape))

    learner = Learner(data_train.shape[0])

    accuracy_data_training = []
    accuracy_data_validation = []
    logger = ConfigLogger()
    logger.describe_model(learner.cNNModel.get_description())

    # Create a local session to run this computation.
    with tf.Session() as tensorflow_session:

        if RESTORE_MODEL:
            # Restore variables from disk.
            learner.saver.restore(tensorflow_session, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            init = tf.global_variables_initializer()
            init_loc = tf.local_variables_initializer()
            tensorflow_session.run(init)
            tensorflow_session.run(init_loc)
            # tf.initialize_all_variables().run()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=tensorflow_session.graph_def)

            print('Initialized!')
            # Loop through training steps.
            print('Total number of iterations = ' + str(int(num_epochs * data_train.shape[0] / BATCH_SIZE)))

            training_indices = range(data_train.shape[0])
            validation_indices = range(data_validation.shape[0])

            for iepoch in range(num_epochs):
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
                    _, l, lr, predictions, _, _, _, _= tensorflow_session.run(
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
                print("\t Accuracy: {:.2%}, Precision: {:.2%}, Recall: {:.2%}, F1: {:.2%}".format(acc, pre, rec, f1s))
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
                print("\t Accuracy: {:.2%}, Precision: {:.2%}, Recall: {:.2%}, F1: {:.2%}".format(acc, pre, rec, f1s))
                print("\t TP: {}, TN: {}, FP: {}, FN: {} \n".format(tp, tn, fp, fn))

                logger.set_validation_score(acc, pre, rec, f1s)

                accuracy_data_validation.append(acc)

                # Save the variables to disk.
                save_path = learner.saver.save(tensorflow_session, FLAGS.train_dir + "/model.ckpt")
                print("\t Model saved in file: %s" % save_path)

        plot_accuracy([accuracy_data_training, accuracy_data_validation], logger.get_timestamp())
        logger.save()

        print("=============================================================")
        print("=============================================================")
        print("")
        #output_training_set_results(tensorflow_session, learner, train_data_filename)



if __name__ == '__main__':
    tf.app.run()
