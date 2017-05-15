from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np

from ops import *

class Finn(object):
    def __init__(self):
        self.df_dim = 2
        self.batch_size = 12
        self.gen_layer_depths = [16, 32, 64, 128]
        self.gen_filter_sizes = [3, 3, 3, 3]


    def discriminator(self, triplet, phase, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()



            h0 = lrelu(conv2d(triplet, self.df_dim, name="d_h0_conv"))
            h1 = lrelu(bn(conv2d(h0, self.df_dim*2, name="d_h1_conv"), phase))
            h2 = lrelu(bn(conv2d(h1, self.df_dim*4, name="d_h2_conv"), phase))
            h3 = lrelu(bn(conv2d(h2, self.df_dim*8, name="d_h3_conv"), phase))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, doublet):
        with tf.variable_scope("generator"):
            conv_outputs = []

            current_input = doublet
            current_inputdepth = doublet.shape[3]
            for i, outputdepth in enumerate(self.gen_layer_depths):
                result = conv_block(current_input, self.gen_filter_sizes[i], outputdepth, name=('g_conv_block'+str(i)) )
                conv_outputs.append(result)
                current_input = result
                current_inputdepth = outputdepth

            z = current_input

            rev_layer_depths = reversed(self.gen_layer_depths)
            rev_filter_sizes = reversed(self.gen_filter_sizes)
            rev_conv_outputs = reversed(conv_outputs)

            # deconv portion
            for i, outputdepth in enumerate(rev_layer_depths[1:]): # reverse process exactly until last step
                result = deconv_block(current_input, rev_filter_sizes[i], current_inputdepth, outputdepth, name=('g_deconv_block'+str(i)) )
                stack = tf.concat([result, rev_conv_outputs[i+1]], 3)
                current_input = stack
                current_inputdepth = 2*outputdepth

            outputdepth = 3 # final image is 3 channel
            return tanh_deconv_layer(current_input, rev_filter_sizes[-1], current_inputdepth, outputdepth, name=('g_tanh_deconv') )


    def build_model(self):
        image_dims = [self.input_height, self.input_width, 6]
        triplet_dims = [self.input_height, self.input_width, 9]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name = 'real_images')
