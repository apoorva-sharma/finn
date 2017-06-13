from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np

from ops import *
from datasets import *
from model import *

class Config:

    def __init__(self, epoch, g_learning_rate, d_learning_rate, beta1, batch_size):
        self.epoch = epoch
        self.g_learning_rate = g_learning_rate
        self.d_learning_rate = d_learning_rate
        self.beta1 = beta1
        self.batch_size = batch_size
        self.checkpoint_dir = './football2_all'
        self.train_gan = True
        self.image_dir = os.path.join(os.getcwd(), 'football2_images')



if __name__ == '__main__':

    cfg = Config(epoch=100, g_learning_rate=.005, d_learning_rate=0.00005, beta1=0.5, batch_size=1)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:

        df_dim = 64.
        batch_size = cfg.batch_size
        dropout_prob = 0.5 # probability of keeping
        l1_weight = 16.
        ssim_weight = 84.
        clipping_weight = 10.
        discriminator_weight = 1.
        writer_path = './l1w_16_ssim_84_cw_10_dw_1'
        video_path = './datasets/football_cif.y4m'
        finn = Finn(sess, df_dim, batch_size, dropout_prob, l1_weight, ssim_weight, clipping_weight, discriminator_weight, writer_path, video_path)
        finn.build_model()
        finn.train(cfg)
        finn.test(cfg)
