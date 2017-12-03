import sys

from helpers.image_helpers import *
from helpers.prediction_helpers import *
from helpers.data_helpers import *
from learner import Learner
from metrics import *
from model import *
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
    train_data = extract_data(train_data_filename, TRAINING_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

    num_epochs = NUM_EPOCHS

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print(len(new_indices))
    print(train_data.shape)
    train_data = train_data[new_indices, :, :, :]
    train_labels = train_labels[new_indices]

    train_size = train_labels.shape[0]
    print(train_size)

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    learner = Learner(train_data, train_size)

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
            print('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))

            training_indices = range(train_size)

            for iepoch in range(num_epochs):
                # Reset local variables, needed for metrics
                tensorflow_session.run(init_loc)

                print("Running epoch {}".format(iepoch))
                # Permute training indices
                perm_indices = numpy.random.permutation(training_indices)

                for step in range(int(train_size / BATCH_SIZE)):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    learner.update_feed_dictionary(batch_data, batch_labels)

                    if step % RECORDING_STEP == 0:

                        summary_str, _, l, lr, predictions, _, _, _, _ = tensorflow_session.run(
                            [summary_op] + learner.get_run_ops() + learner.get_metric_update_ops(),
                            feed_dict=learner.feed_dictionary)
                        # summary_str = s.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                        #print_predictions(predictions, batch_labels)

                        print('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
                        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))
                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions, _, _, _, _= tensorflow_session.run(
                            learner.get_run_ops() + learner.get_metric_update_ops(),
                            feed_dict=learner.get_feed_dictionnary())

                        #print_predictions(predictions, batch_labels)


                tp, fp, tn, fn = tensorflow_session.run(learner.get_metric_ops())
                acc = accuracy(tp, fp, tn, fn)
                pre = precision(tp, fp)
                rec = recall(tp, fn)
                f1s = f1_score(tp, fp, fn)

                print("Accuracy: {:.2%}, Precision: {:.2%}, Recall: {:.2%}, F1: {:.2%}".format(acc, pre, rec, f1s))
                print("TP: {}, TN: {}, FP: {}, FN: {}".format(tp, tn, fp, fn))


                # Save the variables to disk.
                save_path = learner.saver.save(tensorflow_session, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)

        output_training_set_results(tensorflow_session, learner, train_data_filename)



if __name__ == '__main__':
    tf.app.run()
