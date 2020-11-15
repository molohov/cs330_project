# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for maml_vanilla.py."""
from __future__ import print_function

from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
import math
import random
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import opt as contrib_opt
from tensorflow.contrib import image as tf_img
from tensorflow.contrib.layers.python import layers as tf_layers

FLAGS = flags.FLAGS

## Network helpers
def conv_block(x, weight, bias, reuse, scope):
  x = tf.nn.conv2d(x, weight, [1, 1, 1, 1], 'SAME') + bias
  x = tf_layers.batch_norm(
      x, activation_fn=tf.nn.relu, reuse=reuse, scope=scope)
  # # pooling
  # x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
  return x


## Loss functions
def mse(pred, label):
  pred = tf.reshape(pred, [-1])
  label = tf.reshape(label, [-1])
  return tf.reduce_mean(tf.square(pred - label))

def reg1(y_pred, x_train, y_pred_f, x_train_f):
  numerator = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_pred_f), axis=-1)))
  denom = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(x_train - x_train_f), axis=-1)))
  return [numerator, tf.math.minimum(numerator/denom, 1e6)] ## some upper bound


class MAML(object):
  """MAML algo object."""

  def __init__(self, dim_input=1, dim_output=1):
    """must call construct_model() after initializing MAML!"""
    self.dim_input = dim_input
    self.dim_output = dim_output
    self.update_lr = FLAGS.update_lr
    self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
    self.classification = False

    self.loss_func = mse
    self.regularizer = reg1

    self.classification = True
    self.dim_hidden = FLAGS.num_filters
    self.forward = self.forward_conv
    self.construct_weights = self.construct_conv_weights

    self.channels = 1
    self.img_size = int(np.sqrt(self.dim_input / self.channels))

  def construct_model(self,
                      input_tensors=None,
                      prefix='metatrain_',
                      test_num_updates=0):
    """a: training data for inner gradient, b: test data for meta gradient."""

    self.inputa = input_tensors['inputa']
    self.inputb = input_tensors['inputb']
    self.labela = input_tensors['labela']
    self.labelb = input_tensors['labelb']

    with tf.variable_scope('model', reuse=None) as training_scope:
      if 'weights' in dir(self):
        training_scope.reuse_variables()
        weights = self.weights
      else:
        # Define the weights
        self.weights = weights = self.construct_weights()

      # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient
      # updates
      num_updates = max(test_num_updates, FLAGS.num_updates)

      def task_metalearn(inp, reuse=True):
        """Run meta learning."""
        TRAIN = 'train' in prefix  # pylint: disable=invalid-name
        # Perform gradient descent for one task in the meta-batch.
        inputa, inputb, labela, labelb = inp
        task_outputbs, task_lossesb = [], []
        task_msesb = []

        # support_pred and loss, (n_data_per_task, out_dim)
        task_outputa = self.forward(
            inputa, weights, reuse=reuse)  # only not reuse on the first iter
        # labela is (n_data_per_task, out_dim)
        task_lossa = self.loss_func(task_outputa, labela)

        # INNER LOOP (no change with ib)
        grads = tf.gradients(task_lossa, list(weights.values()))
        if FLAGS.stop_grad:
          grads = [tf.stop_gradient(grad) for grad in grads]
        gradients = dict(zip(weights.keys(), grads))
        ## theta_pi = theta - alpha * grads
        fast_weights = dict(
            zip(weights.keys(), [
                weights[key] - self.update_lr * gradients[key]
                for key in weights.keys()
            ]))

        # use theta_pi to forward meta-test
        output = self.forward(inputb, fast_weights, reuse=True)
        task_outputbs.append(output)
        # first meta-test loss, is not necessary to store
        #loss_value = self.loss_func(output, labelb)# + self.loss_func(output_fake, labelb)
        #task_msesb.append(loss_value)
        #task_lossesb.append(loss_value)

        def while_body(fast_weights_values):
          """Update params."""
          loss = self.loss_func(
              self.forward(
                  inputa,
                  dict(zip(fast_weights.keys(), fast_weights_values)),
                  reuse=True), labela)
          grads = tf.gradients(loss, fast_weights_values)
          fast_weights_values = [
              v - self.update_lr * g for v, g in zip(fast_weights_values, grads)
          ]
          return fast_weights_values

        fast_weights_values = tf.while_loop(
            lambda _: True,
            while_body,
            loop_vars=[fast_weights.values()],
            maximum_iterations=num_updates - 1,
            back_prop=TRAIN)
        fast_weights = dict(zip(fast_weights.keys(), fast_weights_values))

        output = self.forward(inputb, fast_weights, reuse=True)
        task_outputbs.append(output)
        task_msesb.append(self.loss_func(output, labelb))

        ## BEGIN second network with FAKE train data/labels

        input_fake = inputa + 1
        label_fake = labela

        random_alter = FLAGS.random_alter
        mirror = FLAGS.mirror
        upside_down = FLAGS.upside_down
        rotate_180 = FLAGS.rotate_180
        inverse = FLAGS.inverse # broken
        random_crop = FLAGS.random_crop # broken
        adjust_contrast = FLAGS.adjust_contrast
        adjust_brightness = FLAGS.adjust_brightness

        if FLAGS.random:
          input_fake = tf.random.uniform(shape=inputa.shape)
          label_fake = tf.random.uniform(shape=labela.shape)

        if random_alter:
          input_fake = tf.math.minimum(1.0, tf.math.maximum(0.0, tf.random.uniform(shape=inputa.shape) - 0.5 + inputa))
          label_fake = label_fake

        if mirror:
          inputa_reshaped = tf.reshape(inputa, [-1, 128, 128])
          inputa_reversed = tf.image.flip_left_right(inputa_reshaped)
          input_fake = tf.reshape(inputa_reversed, [-1, 128*128])
          label_fake = 1 - label_fake

        if upside_down:
          inputa_reshaped = tf.reshape(inputa, [-1, 128, 128])
          inputa_reversed = tf.image.flip_up_down(inputa_reshaped)
          input_fake = tf.reshape(inputa_reversed, [-1, 128*128])
          label_fake = label_fake

        if rotate_180:
          input_reshaped = tf.reshape(inputa, [-1, 128, 128])
          input_upside_down = tf_img.rotate(input_reshaped, math.radians(180))
          input_fake = tf.reshape(input_upside_down, [-1, 128*128])
          label_fake = label_fake

        if inverse:
          input_fake = tf.ones(inputa.shape) - input_fake
          label_fake = label_fake

        if random_crop:
          inputa_reshaped = tf.reshape(inputa, [-1, 128, 128, 1])
          print (f"inputa_reshaped.shape = {inputa_reshaped.shape}")
          inputa_resized = tf.image.resize_images(inputa_reshaped, [256, 256])
          inputa_resized = tf.reshape(inputa_resized, [-1, 256, 256])
          print (f"inputa_resized.shape = {inputa_resized.shape}")
          inputa_cropped = tf.image.random_crop(inputa_resized, [15, 256, 256])
          input_fake = tf.reshape(inputa_cropped, [-1, 128*128])
          label_fake = label_fake

        if adjust_contrast:
          inputa_reshaped = tf.reshape(inputa, [-1, 128, 128, 1])
          inputa_adjusted = tf.image.adjust_contrast(inputa_reshaped, contrast_factor=random.uniform(0, 5))
          input_fake = tf.reshape(inputa_adjusted, [-1, 128*128])
          label_fake = label_fake

        if adjust_brightness:
          inputa_reshaped = tf.reshape(inputa, [-1, 128, 128, 1])
          inputa_adjusted = tf.image.adjust_brightness(inputa_reshaped, delta=random.random())
          input_fake = tf.reshape(inputa_adjusted, [-1, 128*128])
          label_fake = label_fake

        task_output_fake = self.forward(input_fake, weights, reuse=True)  # FIXME use FAKE input
        task_loss_fake = self.loss_func(task_output_fake, label_fake) # FIXME use FAKE label

        # INNER LOOP
        grads = tf.gradients(task_loss_fake, list(weights.values()))
        # we do not want to backprop through these weights
        grads = [tf.stop_gradient(grad) for grad in grads]
        gradients = dict(zip(weights.keys(), grads))
        ## theta_pi = theta - alpha * grads
        fast_weights = dict(
            zip(weights.keys(), [
                weights[key] - self.update_lr * gradients[key]
                for key in weights.keys()
            ]))

        fast_weights_values = tf.while_loop(
            lambda _: True,
            while_body,
            loop_vars=[fast_weights.values()],
            maximum_iterations=num_updates - 1,
            back_prop=TRAIN)
        fast_weights = dict(zip(fast_weights.keys(), fast_weights_values))

        output_fake = self.forward(inputb, fast_weights, reuse=True)
        ## END second network with FAKE train data/labels

        num, reg = self.regularizer(output, inputa, output_fake, input_fake)
        reg_scale = FLAGS.reg_scale
        loss = tf.math.maximum(0.0, self.loss_func(output, labelb) - reg_scale * reg)
        
        task_lossesb.append(loss)
        task_output = [
            task_outputa, task_outputbs, task_lossa, task_lossesb, task_msesb, reg, num
        ]

        return task_output

      if FLAGS.norm is not None:
        # to initialize the batch norm vars, might want to combine this, and
        # not run idx 0 twice.
        _ = task_metalearn(
            (self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]),
            False)

      out_dtype = [
          tf.float32, [tf.float32] * 2, tf.float32, [tf.float32], [tf.float32], tf.float32, tf.float32
      ]
      result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, \
                                                self.labela, self.labelb), dtype=out_dtype, \
                                                parallel_iterations=FLAGS.meta_batch_size)
      outputas, outputbs, lossesa, lossesb, msesb, reg, num = result

    ## Performance & Optimization
    if 'train' in prefix:
      # lossesa is length(meta_batch_size)
      self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(
          FLAGS.meta_batch_size)
      # self.total_losses2 = total_losses2 = [
      #     tf.reduce_sum(msesb[j]) / tf.to_float(FLAGS.meta_batch_size)
      #     for j in range(len(msesb))
      # ]
      self.total_losses2 = total_losses2 = tf.reduce_sum(lossesb) / tf.to_float(FLAGS.meta_batch_size)
      # after the map_fn
      self.outputas, self.outputbs = outputas, outputbs
      # capture the regularizer over the batch
      reg_avg = tf.reduce_sum(reg) / tf.to_float(FLAGS.meta_batch_size)
      tf.summary.scalar(prefix + 'diverse_reg', reg_avg)

      # OUTER LOOP
      if FLAGS.metatrain_iterations > 0:
        if FLAGS.weight_decay:
          optimizer = contrib_opt.AdamWOptimizer(
              weight_decay=FLAGS.beta, learning_rate=self.meta_lr)
        else:
          optimizer = tf.train.AdamOptimizer(self.meta_lr)
        self.gvs_theta = gvs_theta = optimizer.compute_gradients(
            self.total_losses2)
        self.metatrain_op = optimizer.apply_gradients(gvs_theta)

    else:
      self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(
          lossesa) / tf.to_float(FLAGS.meta_batch_size)
      # self.metaval_total_losses2 = total_losses2 = [
      #     tf.reduce_sum(msesb[j]) / tf.to_float(FLAGS.meta_batch_size)
      #     for j in range(len(msesb))
      # ]
      self.metaval_total_losses2 = total_losses2 = tf.reduce_sum(msesb) / tf.to_float(FLAGS.meta_batch_size)

    ## Summaries
    tf.summary.scalar(prefix + 'Pre-mse', total_loss1)
    tf.summary.scalar(prefix + 'Post-mse_' + str(num_updates),
                      total_losses2)
    tf.summary.scalar(prefix + 'numerator', tf.reduce_sum(num) / tf.to_float(FLAGS.meta_batch_size))

  def construct_conv_weights(self):
    """Construct conv weights."""
    weights = {}

    dtype = tf.float32
    conv_initializer = contrib_layers.xavier_initializer_conv2d(dtype=dtype)
    conv_initializer2 = tf.glorot_uniform_initializer(dtype=dtype)
    k = 3
    weights['en_conv1'] = tf.get_variable(
        'en_conv1', [3, 3, 1, 32], initializer=conv_initializer2, dtype=dtype)
    weights['en_bias1'] = tf.Variable(tf.zeros([32]))
    weights['en_conv2'] = tf.get_variable(
        'en_conv2', [3, 3, 32, 48], initializer=conv_initializer2, dtype=dtype)
    weights['en_bias2'] = tf.Variable(tf.zeros([48]))
    weights['en_conv3'] = tf.get_variable(
        'en_conv3', [3, 3, 48, 64], initializer=conv_initializer2, dtype=dtype)
    weights['en_bias3'] = tf.Variable(tf.zeros([64]))

    weights['en_full1'] = tf.get_variable(
        'en_full1', [4096, 196], initializer=conv_initializer2, dtype=dtype)
    weights['en_bias_full1'] = tf.Variable(tf.zeros([196]))

    weights['conv1'] = tf.get_variable(
        'conv1', [k, k, self.channels, self.dim_hidden],
        initializer=conv_initializer,
        dtype=dtype)
    weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
    weights['conv2'] = tf.get_variable(
        'conv2', [k, k, self.dim_hidden, self.dim_hidden],
        initializer=conv_initializer,
        dtype=dtype)
    weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
    weights['conv3'] = tf.get_variable(
        'conv3', [k, k, self.dim_hidden, self.dim_hidden],
        initializer=conv_initializer,
        dtype=dtype)
    weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
    weights['conv4'] = tf.get_variable(
        'conv4', [k, k, self.dim_hidden, self.dim_hidden],
        initializer=conv_initializer,
        dtype=dtype)
    weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))

    weights['w5'] = tf.Variable(
        tf.random_normal([self.dim_hidden, self.dim_output]), name='w5')
    weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
    return weights

  def forward_conv(self, inp, weights, reuse=False, scope=''):
    """Forward conv."""
    # reuse is for the normalization parameters.
    channels = self.channels
    inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])
    en1 = tf.nn.conv2d(inp, weights['en_conv1'], [1, 2, 2, 1],
                       'SAME') + weights['en_bias1']
    en2 = tf.nn.conv2d(en1, weights['en_conv2'], [1, 2, 2, 1],
                       'SAME') + weights['en_bias2']
    pool1 = tf.nn.max_pool(en2, 2, 2, 'VALID')
    en3 = tf.nn.conv2d(pool1, weights['en_conv3'], [1, 2, 2, 1],
                       'SAME') + weights['en_bias3']
    out0 = tf.layers.flatten(en3)
    out1 = tf.nn.relu(
        tf.matmul(out0, weights['en_full1']) + weights['en_bias_full1'])

    out1 = tf.reshape(out1, [-1, 14, 14, 1])

    hidden1 = conv_block(out1, weights['conv1'], weights['b1'], reuse,
                         scope + '0')
    hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse,
                         scope + '1')
    hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse,
                         scope + '2')
    hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse,
                         scope + '3')
    # last hidden layer is 6x6x64-ish, reshape to a vector
    hidden4 = tf.reduce_mean(hidden4, [1, 2])
    # ipdb.set_trace()
    return tf.matmul(hidden4, weights['w5']) + weights['b5']
