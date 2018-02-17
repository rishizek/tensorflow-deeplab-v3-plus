"""Converts PASCAL dataset to TFRecords file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import os
import sys

import PIL.Image
import tensorflow as tf

from utils import dataset_util

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='./dataset/VOCdevkit/VOC2012',
                    help='Path to the directory containing the PASCAL VOC data.')

parser.add_argument('--output_path', type=str, default='./dataset',
                    help='Path to the directory to create TFRecords outputs.')

parser.add_argument('--train_data_list', type=str, default='./dataset/train.txt',
                    help='Path to the file listing the training data.')

parser.add_argument('--valid_data_list', type=str, default='./dataset/val.txt',
                    help='Path to the file listing the validation data.')

parser.add_argument('--image_data_dir', type=str, default='JPEGImages',
                    help='The directory containing the image data.')

parser.add_argument('--label_data_dir', type=str, default='SegmentationClassAug',
                    help='The directory containing the augmented label data.')


def dict_to_tf_example(image_path,
                       label_path):
  """Convert image and label to tf.Example proto.

  Args:
    image_path: Path to a single PASCAL image.
    label_path: Path to its corresponding label.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by image_path is not a valid JPEG or
                if the label pointed to by label_path is not a valid PNG or
                if the size of image does not match with that of label.
  """
  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')

  with tf.gfile.GFile(label_path, 'rb') as fid:
    encoded_label = fid.read()
  encoded_label_io = io.BytesIO(encoded_label)
  label = PIL.Image.open(encoded_label_io)
  if label.format != 'PNG':
    raise ValueError('Label format not PNG')

  if image.size != label.size:
    raise ValueError('The size of image does not match with that of label.')

  width, height = image.size

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/encoded': dataset_util.bytes_feature(encoded_jpg),
    'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    'label/encoded': dataset_util.bytes_feature(encoded_label),
    'label/format': dataset_util.bytes_feature('png'.encode('utf8')),
  }))
  return example


def create_tf_record(output_filename,
                     image_dir,
                     label_dir,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    image_dir: Directory where image files are stored.
    label_dir: Directory where label files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 500 == 0:
      tf.logging.info('On image %d of %d', idx, len(examples))
    image_path = os.path.join(image_dir, example + '.jpg')
    label_path = os.path.join(label_dir, example + '.png')

    if not os.path.exists(image_path):
      tf.logging.warning('Could not find %s, ignoring example.', image_path)
      continue
    elif not os.path.exists(label_path):
      tf.logging.warning('Could not find %s, ignoring example.', label_path)
      continue

    try:
      tf_example = dict_to_tf_example(image_path, label_path)
      writer.write(tf_example.SerializeToString())
    except ValueError:
      tf.logging.warning('Invalid example: %s, ignoring.', example)

  writer.close()


def main(unused_argv):
  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)

  tf.logging.info("Reading from VOC dataset")
  image_dir = os.path.join(FLAGS.data_dir, FLAGS.image_data_dir)
  label_dir = os.path.join(FLAGS.data_dir, FLAGS.label_data_dir)

  if not os.path.isdir(label_dir):
    raise ValueError("Missing Augmentation label directory. "
                     "You may download the augmented labels from the link (Thanks to DrSleep): "
                     "https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip")
  train_examples = dataset_util.read_examples_list(FLAGS.train_data_list)
  val_examples = dataset_util.read_examples_list(FLAGS.valid_data_list)

  train_output_path = os.path.join(FLAGS.output_path, 'voc_train.record')
  val_output_path = os.path.join(FLAGS.output_path, 'voc_val.record')

  create_tf_record(train_output_path, image_dir, label_dir, train_examples)
  create_tf_record(val_output_path, image_dir, label_dir, val_examples)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
