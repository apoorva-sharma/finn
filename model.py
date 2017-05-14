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

    def build_model(self):
        image_dims = [self.input_height, self.input_width, 6]
        triplet_dims = [self.input_height, self.input_width, 9]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name = 'real_images')