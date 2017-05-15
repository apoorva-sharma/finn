import math
import numpy as np
import tensorflow as tf


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def conv2d(input_, output_dim, hh=3, ww=3, stride_h =1, stride_w=1, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [hh, ww, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, stride_h, stride_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv,biases), conv.get_shape())

        return conv

def conv_block(input_, filter_size, output_depth, name="conv_block"):
    with tf.variable_scope(name):
        h1 = lrelu( conv2d(input_, output_depth, filter_size, filter_size, name="conv2d1") )
        h2 = lrelu( conv2d(h1, output_depth, filter_size, filter_size, name="conv2d2") )

        return tf.nn.max_pool(h2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", data_format="NHWC")

def deconv_block(input_, filter_size, output_depth, name="deconv_block"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [2, 2, output_depth, input_.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_shape = [ int(input_.shape[0]), 2*int(input_.shape[1]), 2*int(input_.shape[2]), output_depth]
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, 2, 2, 1], padding='VALID')
        biases = tf.get_variable('biases', [output_depth], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv,biases), deconv.get_shape())
        print('\tdeconv:',deconv.get_shape())
        h2 = lrelu( conv2d(deconv, output_depth, filter_size, filter_size, name="conv2d1") )
        print('\th2:',h2.get_shape())

        return h2

def tanh_deconv_block(input_, filter_size, output_depth, name="deconv_block"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [2, 2, output_depth, input_.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_shape = [ int(input_.shape[0]), 2*int(input_.shape[1]), 2*int(input_.shape[2]), output_depth]
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_depth], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv,biases), deconv.get_shape())

        h2 = tf.nn.tanh( conv2d(deconv, output_depth, filter_size, filter_size, name="conv2d1") )

        return h2



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
