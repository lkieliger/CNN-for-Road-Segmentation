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
    for filename in os.listdir(TRAIN_DATA_TRAIN_SPLIT_IMAGES_PATH):
        #pimg = get_prediction_with_groundtruth(train_data_filename, i, learner.cNNModel, session)
        #Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
        oimg = get_prediction_with_overlay(os.path.join(TRAIN_DATA_TRAIN_SPLIT_IMAGES_PATH, filename), learner.cNNModel, session)
        oimg.save(prediction_training_dir + "overlay_" + filename)


def output_validation_set_results(session, learner, train_data_filename):
    print("Running prediction on validation set")
    prediction_training_dir = "predictions_training/"
    if not os.path.isdir(prediction_training_dir):
        os.mkdir(prediction_training_dir)
    for filename in os.listdir(TRAIN_DATA_VALIDATION_SPLIT_IMAGES_PATH):
        #pimg = get_prediction_with_groundtruth(train_data_filename, i, learner.cNNModel, session)
        #Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
        oimg = get_prediction_with_overlay(os.path.join(TRAIN_DATA_VALIDATION_SPLIT_IMAGES_PATH, filename), learner.cNNModel, session)
        oimg.save(prediction_training_dir + "overlay_val_" + filename)

def main(argv=None):  # pylint: disable=unused-argument

    data_dir = 'data/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'

    permutations = np.random.permutation(range(NUM_IMAGES))

    """
    # Extract it into numpy arrays.
    data = extract_data(train_data_filename, permutations)
    labels = extract_labels(train_labels_filename, permutations)

    print("Data shape: {}".format(data.shape))

    num_epochs = NUM_EPOCHS

    c0 = 0
    c1 = 0
    for i in range(len(labels)):
        if labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    # Shuffling test data
    #np.random.seed(SEED)
    #shuffling_indices = np.random.permutation(range(data.shape[0]))
    #data = data[shuffling_indices, :, :, :]
    #labels = labels[shuffling_indices]
    
    """
    data_train = extract_data(TRAIN_DATA_TRAIN_SPLIT_IMAGES_PATH)
    data_validation = extract_data(TRAIN_DATA_VALIDATION_SPLIT_IMAGES_PATH)
    data_test = extract_data(TRAIN_DATA_TEST_SPLIT_IMAGES_PATH)

    labels_train = extract_labels(TRAIN_DATA_TRAIN_SPLIT_GROUNDTRUTH_PATH)
    labels_validation = extract_labels(TRAIN_DATA_VALIDATION_SPLIT_GROUNDTRUTH_PATH)
    labels_test = extract_labels(TRAIN_DATA_TEST_SPLIT_GROUNDTRUTH_PATH)

    if BALANCE_DATA:
        data_train, labels_train = balance_dataset(data_train, labels_train)
        data_validation, labels_validation = balance_dataset(data_validation, labels_validation)

    # Split data
    #data_train, data_validation, data_test, labels_train, labels_validation, labels_test = split_patches(data, labels)

    print("Training data shape: {}".format(data_train.shape))
    print("Validation data shape: {}".format(data_validation.shape))
    print("Test data shape: {}".format(data_test.shape))

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
            learner.saver.restore(tensorflow_session, FLAGS.train_dir + "/model.ckpt")
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

            weigths_1 = tensorflow_session.run(learner.cNNModel.conv1_weights)
            plot_conv_weights(weigths_1, logger.get_timestamp())

        print("=============================================================")
        print("=============================================================")
        print("")

        output_training_set_results(tensorflow_session, learner, train_data_filename)
        output_validation_set_results(tensorflow_session, learner, train_data_filename)




if __name__ == '__main__':
    tf.app.run()
