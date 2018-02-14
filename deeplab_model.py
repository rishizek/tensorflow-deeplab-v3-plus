"""DeepLab v3 models based on slim library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.slim.python.slim.nets import resnet_utils

from utils import preprocessing


def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
  """Atrous Spatial Pyramid Pooling.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    is_training: A boolean denoting whether the input is for training.
    depth: The depth of the ResNet unit output.

  Returns:
    The atrous spatial pyramid pooling output.
  """
  with tf.variable_scope("assp"):
    if output_stride not in [8, 16]:
      raise ValueError('output_stride must be either 8 or 16.')

    atrous_rates = [6, 12, 18]
    if output_stride == 8:
      atrous_rates = [2*rate for rate in atrous_rates]

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      with arg_scope([layers.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]
        # (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
        conv_3x3_1 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
        conv_3x3_2 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
        conv_3x3_3 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

        # (b) the image-level features
        with tf.variable_scope("image_level_features"):
          # global average pooling
          image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
          # 1×1 convolution with 256 filters( and batch normalization)
          image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
          # bilinearly upsample features
          image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

        return net


def deeplab_v3_generator(num_classes,
                         output_stride,
                         base_architecture,
                         pre_trained_model,
                         batch_norm_decay,
                         data_format='channels_last'):
  """Generator for DeepLab v3 models.

  Args:
    num_classes: The number of possible classes for image classification.
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    base_architecture: The architecture of base Resnet building block.
    pre_trained_model: The path to the directory that contains pre-trained models.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
      Only 'channels_last' is supported currently.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the DeepLab v3 model.
  """
  if data_format is None:
    # data_format = (
    #     'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    pass

  if base_architecture not in ['resnet_v2_50', 'resnet_v2_101']:
    raise ValueError("'base_architrecture' must be either 'resnet_v2_50' or 'resnet_v2_50'.")

  if base_architecture == 'resnet_v2_50':
    base_model = resnet_v2.resnet_v2_50
  else:
    base_model = resnet_v2.resnet_v2_101

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    tf.logging.info('net shape: {}'.format(inputs.shape))

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      logits, end_points = base_model(inputs,
                                      num_classes,
                                      is_training=is_training,
                                      global_pool=False,
                                      output_stride=output_stride)

    if is_training:
      exclude = [base_architecture + '/logits', 'global_step']
      variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
      tf.train.init_from_checkpoint(pre_trained_model,
                                    {v.name.split(':')[0]: v for v in variables_to_restore})

    inputs_size = tf.shape(inputs)[1:3]
    net = end_points[base_architecture + '/block4']
    net = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)
    with tf.variable_scope("upsampling_logits"):
      net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
      logits = tf.image.resize_bilinear(net, inputs_size, name='upsample')

    return logits

  return model


def deeplabv3_model_fn(features, labels, mode, params):
  """Model function for PASCAL VOC."""
  images_summary = tf.cast(
      tf.map_fn(lambda x: preprocessing.mean_image_addition(
          x,
          [params['r_mean'], params['g_mean'], params['b_mean']]),
          features),
      tf.uint8)

  network = deeplab_v3_generator(params['num_classes'],
                                 params['output_stride'],
                                 params['base_architecture'],
                                 params['pre_trained_model'],
                                 params['batch_norm_decay'])

  logits = network(features, mode == tf.estimator.ModeKeys.TRAIN)

  predictions = {
      'classes': tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  labels_summary = tf.py_func(preprocessing.decode_labels,
                              [labels, params['batch_size'], params['num_classes']], tf.uint8)
  preds_summary = tf.py_func(preprocessing.decode_labels,
                             [predictions['classes'], params['batch_size'], params['num_classes']],
                             tf.uint8)

  tf.summary.image('images',
                   tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
                   max_outputs=params['tensorboard_images_max_outputs'])  # Concatenate row-wise.

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  labels = tf.squeeze(labels, axis=3)  # reduce the channel dimension.

  logits_by_num_classes = tf.reshape(logits, [-1, params['num_classes']])
  labels_flat = tf.reshape(labels, [-1, ])

  valid_indices = tf.to_int32(labels_flat <= params['num_classes'] - 1)
  valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
  valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=valid_logits, labels=valid_labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss.
  with tf.variable_scope("total_loss"):
    loss = cross_entropy + params['weight_decay'] * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  # loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    if params['learning_rate_policy'] == 'piecewise':
      # Scale the learning rate linearly with the batch size. When the batch size
      # is 128, the learning rate should be 0.1.
      initial_learning_rate = 0.1 * params['batch_size'] / 128
      batches_per_epoch = params['num_train'] / params['batch_size']
      # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
      boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
      values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
      learning_rate = tf.train.piecewise_constant(
          tf.cast(global_step, tf.int32), boundaries, values)
    elif params['learning_rate_policy'] == 'poly':
      learning_rate = tf.train.polynomial_decay(
          params['initial_learning_rate'], tf.cast(global_step, tf.int32),
          params['max_iter'], params['end_learning_rate'], power=params['power'])
    else:
      raise ValueError('Learning rate policy must be "piecewise" or "poly"')

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=params['momentum'])

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
  mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, params['num_classes'])
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