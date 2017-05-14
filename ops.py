import math
import numpy as np
import tensorflow as tf


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def conv2d(input_, output_dim, hh=3, ww=3, stride_h =1, stride_w=1, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [hh, ww, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, stride_h, stride_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv,biases), conv.get_shape())

        return conv

def bn(x, phase, center=True, scale=True, name = 'batch_norm'):
    return tf.contrib.layers.batch_norm(inputs = x, center=center, scale=scale,
                                        is_training=phase, scope=name, data_format = 'NHWC')

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias