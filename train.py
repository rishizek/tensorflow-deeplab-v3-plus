"""Runs a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import deeplab_model
from utils import preprocessing
from tensorflow.python import debug as tf_debug

import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='./model',
                    help='Base directory for the model.')

parser.add_argument('--clean_model_dir', action='store_true',
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--train_epochs', type=int, default=20,
                    help='Number of training epochs. '
                         'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582).')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of training epochs to run between evaluations.')

parser.add_argument('--save_num_images', type=int, default=6,
                    help='How many images to save.')

parser.add_argument('--batch_size', type=int, default=6,
                    help='Number of examples per batch.')

parser.add_argument('--learning_rate_policy', type=str, default='poly',
                    choices=['poly', 'piecewise'],
                    help='Learning rate policy to optimize loss.')

parser.add_argument('--max_iter', type=int, default=30001,
                    help='Number of maximum iteration used for "poly" learning rate policy.')

parser.add_argument('--data_dir', type=str, default='./dataset/',
                    help='Path to the directory containing the PASCAL VOC data tf record.')

parser.add_argument('--image_data_dir', type=str, default='JPEGImages',
                    help='The directory containing the image data.')

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--pre_trained_model', type=str, default='./ini_checkpoints/resnet_v2_101/resnet_v2_101.ckpt',
                    help='Path to the pre-trained model checkpoint.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_NUM_CLASSES = 21
_HEIGHT = 513
_WIDTH = 513
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

_STARTER_LEARNING_RATE = 7e-3
_END_LEARNING_RATE = 1e-8
_POWER = 0.9
_WEIGHT_DECAY = 5e-4
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train': 10582,
    'validation': 1449,
}


def get_filenames(is_training, data_dir):
  """Return a list of filenames.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: path to the the directory containing the input data.

  Returns:
    A list of file names.
  """
  if is_training:
    return [os.path.join(data_dir, 'voc_train.record')]
  else:
    return [os.path.join(data_dir, 'voc_val.record')]


def parse_record(raw_record):
  """Parse PASCAL image and label from a tf record."""
  keys_to_features = {
      'image/height':
      tf.FixedLenFeature((), tf.int64),
      'image/width':
      tf.FixedLenFeature((), tf.int64),
      'image/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
      tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'label/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'label/format':
      tf.FixedLenFeature((), tf.string, default_value='png'),
  }

  parsed = tf.parse_single_example(raw_record, keys_to_features)

  # height = tf.cast(parsed['image/height'], tf.int32)
  # width = tf.cast(parsed['image/width'], tf.int32)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]), _DEPTH)
  image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
  image.set_shape([None, None, 3])

  label = tf.image.decode_image(
      tf.reshape(parsed['label/encoded'], shape=[]), 1)
  label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
  label.set_shape([None, None, 1])

  return image, label


def preprocess_image(image, label, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Randomly scale the image and label.
    image, label = preprocessing.random_rescale_image_and_label(
        image, label, _MIN_SCALE, _MAX_SCALE)

    # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
    image, label = preprocessing.random_crop_or_pad_image_and_label(
        image, label, _HEIGHT, _WIDTH, _IGNORE_LABEL)

    # Randomly flip the image and label horizontally.
    image, label = preprocessing.random_flip_left_right_image_and_label(
        image, label)

  image = preprocessing.mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

  return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
  dataset = dataset.flat_map(tf.data.TFRecordDataset)

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  dataset = dataset.map(parse_record)
  dataset = dataset.map(
      lambda image, label: preprocess_image(image, label, is_training))
  dataset = dataset.prefetch(batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels


def deeplabv3_model_fn(features, labels, mode, params):
  """Model function for PASCAL VOC."""
  images_summary = tf.cast(
      tf.map_fn(lambda x: preprocessing.mean_image_addition(x, [_R_MEAN, _G_MEAN, _B_MEAN]), features),
      tf.uint8)

  network = deeplab_model.deeplab_v3_generator(_NUM_CLASSES, params['output_stride'],
                                               params['base_architecture'],
                                               params['pre_trained_model'])

  logits = network(features, mode == tf.estimator.ModeKeys.TRAIN)

  predictions = {
      'classes': tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  labels_summary = tf.py_func(preprocessing.decode_labels,
                              [labels, params['batch_size'], _NUM_CLASSES], tf.uint8)
  preds_summary = tf.py_func(preprocessing.decode_labels,
                             [predictions['classes'], params['batch_size'], _NUM_CLASSES],
                             tf.uint8)

  tf.summary.image('images',
                   tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
                   max_outputs=FLAGS.save_num_images)  # Concatenate row-wise.

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  labels = tf.squeeze(labels, axis=3)  # reduce the channel dimension.

  logits_by_num_classes = tf.reshape(logits, [-1, _NUM_CLASSES])
  labels_flat = tf.reshape(labels, [-1, ])

  valid_indices = tf.to_int32(labels_flat <= _NUM_CLASSES - 1)
  valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
  valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=valid_logits, labels=valid_labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss.
  with tf.variable_scope("total_loss"):
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  # loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    if FLAGS.learning_rate_policy == 'piecewise':
      # Scale the learning rate linearly with the batch size. When the batch size
      # is 128, the learning rate should be 0.1.
      initial_learning_rate = 0.1 * params['batch_size'] / 128
      batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
      # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
      boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
      values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
      learning_rate = tf.train.piecewise_constant(
          tf.cast(global_step, tf.int32), boundaries, values)
    elif FLAGS.learning_rate_policy == 'poly':
      learning_rate = tf.train.polynomial_decay(
          _STARTER_LEARNING_RATE, tf.cast(global_step, tf.int32),
          FLAGS.max_iter, _END_LEARNING_RATE, power=_POWER, cycle=True)
    else:
      raise ValueError('Learning rate policy must be "piecewise" or "poly"')

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None

  preds_flat = tf.reshape(predictions['classes'], [-1, ])
  valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
  accuracy = tf.metrics.accuracy(
      valid_labels, valid_preds)
  mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, _NUM_CLASSES)
  metrics = {'px_accuracy': accuracy, 'mean_iou': mean_iou}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_px_accuracy')
  tf.summary.scalar('train_px_accuracy', accuracy[1])

  def compute_mean_accuracy(total_cm):
    """Compute the mean per class accuracy via the confusion matrix."""
    per_row_sum = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = per_row_sum

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0), denominator,
        tf.ones_like(denominator))
    accuracies = tf.div(cm_diag, denominator)
    return tf.reduce_mean(accuracies)

  tf.identity(compute_mean_accuracy(mean_iou[1]), name='train_mean_iou')
  tf.summary.scalar('train_mean_iou', compute_mean_accuracy(mean_iou[1]))

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  if FLAGS.clean_model_dir:
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
  model = tf.estimator.Estimator(
      model_fn=deeplabv3_model_fn, model_dir=FLAGS.model_dir, config=run_config,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': FLAGS.batch_size,
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': FLAGS.pre_trained_model
      })

  for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    tensors_to_log = {
      'learning_rate': 'learning_rate',
      'cross_entropy': 'cross_entropy',
      'train_px_accuracy': 'train_px_accuracy',
      'train_mean_iou': 'train_mean_iou',
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    train_hooks = [logging_hook]
    eval_hooks = None

    if FLAGS.debug:
      debug_hook = tf_debug.LocalCLIDebugHook()
      train_hooks.append(debug_hook)
      eval_hooks = [debug_hook]

    model.train(
        input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
        hooks=train_hooks,
        # steps=1  # For debug
    )

    # Evaluate the model and print results
    eval_results = model.evaluate(
        # Batch size must be 1 for testing because the images' size differs
        input_fn=lambda: input_fn(False, FLAGS.data_dir, 1),
        hooks=eval_hooks,
        # steps=1  # For debug
    )
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
