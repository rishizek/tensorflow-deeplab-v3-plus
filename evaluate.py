"""Evaluate a DeepLab v3 model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import deeplab_model
from utils import preprocessing
from utils import dataset_util

import numpy as np
import timeit

parser = argparse.ArgumentParser()

parser.add_argument('--image_data_dir', type=str, default='dataset/VOCdevkit/VOC2012/JPEGImages',
                    help='The directory containing the image data.')

parser.add_argument('--label_data_dir', type=str, default='dataset/VOCdevkit/VOC2012/SegmentationClassAug',
                    help='The directory containing the ground truth label data.')

parser.add_argument('--evaluation_data_list', type=str, default='./dataset/val.txt',
                    help='Path to the file listing the evaluation images.')

parser.add_argument('--model_dir', type=str, default='./model',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

_NUM_CLASSES = 21


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  examples = dataset_util.read_examples_list(FLAGS.evaluation_data_list)
  image_files = [os.path.join(FLAGS.image_data_dir, filename) + '.jpg' for filename in examples]
  label_files = [os.path.join(FLAGS.label_data_dir, filename) + '.png' for filename in examples]

  features, labels = preprocessing.eval_input_fn(image_files, label_files)

  predictions = deeplab_model.deeplabv3_plus_model_fn(
      features,
      labels,
      tf.estimator.ModeKeys.EVAL,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': None,
          'batch_norm_decay': None,
          'num_classes': _NUM_CLASSES,
          'freeze_batch_norm': True
      }).predictions

  # Manually load the latest checkpoint
  saver = tf.train.Saver()
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Loop through the batches and store predictions and labels
    step = 1
    sum_cm = np.zeros((_NUM_CLASSES, _NUM_CLASSES), dtype=np.int32)
    start = timeit.default_timer()
    while True:
      try:
        preds = sess.run(predictions)
        sum_cm += preds['confusion_matrix']
        if not step % 100:
          stop = timeit.default_timer()
          tf.logging.info("current step = {} ({:.3f} sec)".format(step, stop-start))
          start = timeit.default_timer()
        step += 1
      except tf.errors.OutOfRangeError:
        break

    def compute_mean_iou(total_cm):
      """Compute the mean intersection-over-union via the confusion matrix."""
      sum_over_row = np.sum(total_cm, axis=0).astype(float)
      sum_over_col = np.sum(total_cm, axis=1).astype(float)
      cm_diag = np.diagonal(total_cm).astype(float)
      denominator = sum_over_row + sum_over_col - cm_diag

      # The mean is only computed over classes that appear in the
      # label or prediction tensor. If the denominator is 0, we need to
      # ignore the class.
      num_valid_entries = np.sum((denominator != 0).astype(float))

      # If the value of the denominator is 0, set it to 1 to avoid
      # zero division.
      denominator = np.where(
          denominator > 0,
          denominator,
          np.ones_like(denominator))

      ious = cm_diag / denominator

      print('Intersection over Union for each class:')
      for i, iou in enumerate(ious):
        print('    class {}: {:.4f}'.format(i, iou))

      # If the number of valid entries is 0 (no classes) we return 0.
      m_iou = np.where(
          num_valid_entries > 0,
          np.sum(ious) / num_valid_entries,
          0)
      m_iou = float(m_iou)
      print('mean Intersection over Union: {:.4f}'.format(float(m_iou)))

    def compute_accuracy(total_cm):
      """Compute the accuracy via the confusion matrix."""
      denominator = total_cm.sum().astype(float)
      cm_diag_sum = np.diagonal(total_cm).sum().astype(float)

      # If the number of valid entries is 0 (no classes) we return 0.
      accuracy = np.where(
          denominator > 0,
          cm_diag_sum / denominator,
          0)
      accuracy = float(accuracy)
      print('Pixel Accuracy: {:.4f}'.format(float(accuracy)))

    compute_mean_iou(sum_cm)
    compute_accuracy(sum_cm)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
