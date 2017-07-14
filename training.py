import os
from datetime import datetime
import time
import argparse
import tensorflow as tf

import network
import inputs

FLAGS = tf.app.flags.FLAGS

# Basic I/O parameters
tf.app.flags.DEFINE_string('train_dir',os.path.join(os.getcwd(),'train'),
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
# Basic network parameters.
tf.app.flags.DEFINE_integer('batch_size', 96,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 1000000,
                            """Number of Data Epochs to do training""")
tf.app.flags.DEFINE_integer('NUM_EXAMPLES_PER_EPOCH', 50000,
                            """Number of classes in training/evaluation data.""")

# Basic parameters describing the data set.
tf.app.flags.DEFINE_integer('NUM_CLASSES', 27,
                            """Number of classes in training/evaluation data.""")
tf.app.flags.DEFINE_integer('IMAGE_HEIGHT', 85,
                            """IMAGE HEIGHT""")
tf.app.flags.DEFINE_integer('IMAGE_WIDTH', 120,
                            """IMAGE WIDTH""")
tf.app.flags.DEFINE_integer('IMAGE_DEPTH', 1,
                            """IMAGE DEPTH""")
tf.app.flags.DEFINE_integer('CROP_HEIGHT', 58,
                            """CROP HEIGHT""")
tf.app.flags.DEFINE_integer('CROP_WIDTH', 79,
                            """CROP WIDTH""")

def train():
    """
    Train the network for a number of steps
    Args:
        images_path: file path to images
        labels_path: file path to labels

    Returns: Nothing

    """
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        with tf.variable_scope('Input') as scope:
            # Add queue runner to the graph
            filename_queue = tf.train.string_input_producer(['../multi_slice_GP_training/'+
                                                             'oxide_tilts_GP_train_171x240_3deg.tfrecords'],
                                                            num_epochs=FLAGS.num_epochs)
            # pass the filename_queue to the input class to decode
            dset = inputs.Dataset_TFRecords(filename_queue,FLAGS)
            image, label = dset.decode_image_label()

            # distort images and generate examples batch
            images, labels = dset.train_images_labels_batch(image, label, noise_min=0.0,
                                                            noise_max=0.1,
                                                            random_glimpses=True)


        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = network.inference(images,FLAGS)

        # Calculate loss.
        loss = network.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = network.train(loss, global_step, FLAGS)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                       examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
            config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
                while not mon_sess.should_stop():
                    mon_sess.run(train_op)

def main(argv):  # pylint: disable=unused-argument
    # parser = argparse.ArgumentParser(description='Process images and labels path')
    # parser.add_argument('Path to data',type=str,help='path to 4d images numpy array')
    # parser.add_argument ('Path to labels', type=str, help='path to 2d labels numpy array')
    # args = parser.parse_args()
  # if tf.gfile.Exists(FLAGS.train_dir):
  #   tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
