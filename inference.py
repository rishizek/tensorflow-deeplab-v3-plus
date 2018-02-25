"""Run inference a DeepLab v3 model using tf.estimator API."""

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

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='dataset/VOCdevkit/VOC2012/JPEGImages',
                    help='The directory containing the image data.')

parser.add_argument('--output_dir', type=str, default='./dataset/inference_output',
                    help='Path to the directory to generate the inference results')

parser.add_argument('--infer_data_list', type=str, default='./dataset/sample_images_list.txt',
                    help='Path to the file listing the inferring images.')

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

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_NUM_CLASSES = 21


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  pred_hooks = None
  if FLAGS.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    pred_hooks = [debug_hook]

  model = tf.estimator.Estimator(
      model_fn=deeplab_model.deeplabv3_plus_model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': None,
          'batch_norm_decay': None,
          'num_classes': _NUM_CLASSES,
      })

  examples = dataset_util.read_examples_list(FLAGS.infer_data_list)
  image_files = [os.path.join(FLAGS.data_dir, filename) for filename in examples]

  predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn(image_files),
        hooks=pred_hooks)

  output_dir = FLAGS.output_dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  for pred_dict, image_path in zip(predictions, image_files):
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = image_basename + '_mask.png'
    path_to_output = os.path.join(output_dir, output_filename)

    print("generating:", path_to_output)
    mask = pred_dict['decoded_labels']
    mask = Image.fromarray(mask)
    plt.axis('off')
    plt.imshow(mask)
    plt.savefig(path_to_output, bbox_inches='tight')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
