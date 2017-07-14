from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
import network
import os
import inputs

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir',os.path.join(os.getcwd(),'eval'),
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', os.path.join(os.getcwd(),'train'),
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 50000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")

# Basic network parameters.
tf.app.flags.DEFINE_integer('batch_size', 256,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 1000000,
                            """Number of Data Epochs to do training""")


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

# Basic training/evaluation data parameters Global constants describing the data set.
tf.app.flags.DEFINE_integer('NUM_EXAMPLES_PER_EPOCH', 50000,
                            """Number of classes in training/evaluation data.""")


def eval_once(saver, summary_writer, top_k_op, categories, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    categories: classes in batch.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      sorted_predictions = np.zeros(shape=(FLAGS.NUM_CLASSES,))
      while step < num_iter and not coord.should_stop():
        # sum up all predictions regardless of class
        predictions = np.array(sess.run([top_k_op])).flatten()
        true_count += np.sum(predictions)
        # sum up predictions per class
        classes = np.array(sess.run([categories])).flatten()
        uniq_cls, uniq_indx, uniq_cts = np.unique(classes, return_index=True, return_counts=True)
        zeroes = np.zeros_like(sorted_predictions)
        zeroes[uniq_cls] = predictions[uniq_indx]
        sorted_predictions += zeroes
        # increment step
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      # Load class names and compute precision per class
      tilt_patterns = np.load('../multi_slice_GP_arrays/tilt_patterns_GP.npy')
      categories = np.unique(tilt_patterns)
      precision_per_catg = sorted_predictions/step
      dic = dict([(catg, np.round(prec,2)) for catg, prec in zip(categories,precision_per_catg)])
      print('%s: precision per class @ 1' % datetime.now())
      for key in dic.iterkeys():
          print("%s : %2.2f"%(key,dic[key]))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval Data for a number of steps."""
  with tf.Graph().as_default() as g:
      with tf.variable_scope('Input_test') as scope:
          # Add queue runner to the graph
          filename_queue = tf.train.string_input_producer(['../multi_slice_GP_training/'+
                                                           'oxide_tilts_GP_test_171x240_bin2.tfrecords'])
                                                          # num_epochs=FLAGS.num_epochs)
          # pass the filename_queue to the input class to decode
          dset = inputs.Dataset_TFRecords(filename_queue, FLAGS)
          image, label = dset.decode_image_label()

          # distort images and generate examples batch
          images, labels = dset.eval_images_labels_batch(image, label, noise_min= 0.01, noise_max=0.05,
                                                         random_glimpses=True)

      # Build a Graph that computes the logits predictions from the inference model.
      logits = network.inference(images, FLAGS)
      #print('Logits shape: %s' %(format(logits.shape)))
      #print('Labels shape: %s' % (format(labels.shape)))
      labels = tf.argmax(labels, axis=1)
      #print('Reshaped Labels shape: %s' % (format(labels.shape)))

      # categories = tf.argmax(labels, axis=0)

      # Calculate predictions.
      in_top_k_op = tf.nn.in_top_k(logits, labels, 1)
      # top_k_op = tf.nn.top_k(logits,labels)

      # Restore the moving average version of the learned variables for eval.
      variable_averages = tf.train.ExponentialMovingAverage(
        network.MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()
      saver = tf.train.Saver(variables_to_restore)

      # Build the summary operation based on the TF collection of Summaries.
      summary_op = tf.summary.merge_all()

      summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

      while True:
        eval_once(saver, summary_writer, tf.cast(in_top_k_op,tf.float32), labels, summary_op)
        if FLAGS.run_once:
            break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()