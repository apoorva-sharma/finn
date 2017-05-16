from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from PIL import Image

from ops import *
from datasets import *

class Finn(object):
    def __init__(self, sess, df_dim, batch_size, dropout_prob, l1_weight, writer_path, video_path):
        self.df_dim = df_dim
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.l1_weight = l1_weight

        self.sess = sess
        self.writer_path = writer_path
        self.filename = 'yomama'
        self.video_path = video_path


        data = generateDataSet(self.video_path)
        self.train_doublets = data["train_doublets"]
        self.train_triplets = data["train_triplets"]
        self.train_singlets = self.train_triplets[:,:,:,3:6]
        self.val_doublets = data["val_doublets"]
        self.val_targets = data["val_targets"]
        self.mean_img = data["mean_img"]

        self.input_height = self.train_doublets.shape[1]
        self.input_width = self.train_doublets.shape[2]
        self.dataset_name = 'abc'

        self.gen_layer_depths = [16, 32, 64, 128]
        self.gen_filter_sizes = [3, 3, 3, 3]

        self.max_outputs = 100

    def discriminator(self, triplet, phase, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(triplet, self.df_dim, name="d_h0_conv"))
            h1 = lrelu(bn(conv2d(h0, self.df_dim*2, name="d_h1_conv"), phase, name="d_h1_bn"))
            h2 = lrelu(bn(conv2d(h1, self.df_dim*4, name="d_h2_conv"), phase, name="d_h2_bn"))
            h3 = lrelu(bn(conv2d(h2, self.df_dim*8, name="d_h3_conv"), phase, name="d_h3_bn"))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, doublet):
        with tf.variable_scope("generator"):
            conv_outputs = []

            current_input = doublet
            current_inputdepth = doublet.shape[3]
            for i, outputdepth in enumerate(self.gen_layer_depths):
                result = conv_block(current_input, self.is_training, self.gen_filter_sizes[i], outputdepth, name=('g_conv_block'+str(i)) )
                conv_outputs.append(result)
                current_input = result
                current_inputdepth = outputdepth

            z = current_input

            rev_layer_depths = list(reversed(self.gen_layer_depths))
            rev_filter_sizes = list(reversed(self.gen_filter_sizes))
            rev_conv_outputs = list(reversed(conv_outputs))

            # deconv portion
            for i, outputdepth in enumerate(rev_layer_depths[1:]): # reverse process exactly until last step

                result = deconv_block(current_input, self.is_training, rev_filter_sizes[i], outputdepth, name=('g_deconv_block'+str(i)) )
                if i <= 4:
                    result = tf.nn.dropout(result, self.dropout_prob)
                print( i, result.get_shape() )
                stack = tf.concat([result, rev_conv_outputs[i+1]], 3)
                # print( i, stack.get_shape() )
                current_input = stack

            outputdepth = 3 # final image is 3 channel
            return tanh_deconv_block(current_input, self.is_training, rev_filter_sizes[-1], outputdepth, name=('g_tanh_deconv') )


    def build_model(self):
        singlet_dims = [self.input_height, self.input_width, 3]
        image_dims = [self.input_height, self.input_width, 6]
        triplet_dims = [self.input_height, self.input_width, 9]

        self.mean_placeholder = tf.placeholder(tf.float32, singlet_dims, name = 'mean_img')
        self.singlets = tf.placeholder(tf.float32, [self.batch_size] + singlet_dims, name = 'singlets')
        self.doublets = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name = 'doublets')
        self.triplets = tf.placeholder(tf.float32, [self.batch_size] + triplet_dims, name = 'triplets')
        self.is_training = tf.placeholder(tf.bool, (), name = 'is_training')

        self.G = self.generator(self.doublets)
        eps = 1e-5
        self.g_loss_l1 = tf.reduce_mean(tf.sqrt(tf.square(self.G - self.singlets) + eps))

        self.D_real, self.D_real_logits = self.discriminator(self.triplets, self.is_training, reuse=False)

        self.before, self.after = tf.split(self.doublets, [3, 3], 3)
        self.fake_triplets = tf.concat([self.before, self.G, self.after], 3)

        self.D_fake, self.D_fake_logits = self.discriminator(self.fake_triplets, self.is_training, reuse=True)

        self.d_real_sum = tf.summary.histogram("d_real", self.D_real)
        self.d_fake_sum = tf.summary.histogram("d_fake", self.D_fake)
        self.num_images = self.batch_size
        self.G_image = tf.summary.image("G", self.G + self.mean_img, max_outputs=self.max_outputs)
        self.before_image = tf.summary.image("Z1", self.before + self.mean_img, max_outputs=self.max_outputs)
        self.after_image = tf.summary.image("Z2", self.after + self.mean_img, max_outputs=self.max_outputs)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones_like(self.D_real)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake)))

        self.g_loss_total = tf.add(-self.d_loss_fake, self.l1_weight*self.g_loss_l1)

        self.d_loss_sum_real = tf.summary.scalar("real_loss", self.d_loss_real)
        self.d_loss_sum_fake = tf.summary.scalar("fake_loss", self.d_loss_fake)

        self.d_loss = self.d_loss_fake + self.d_loss_real

        self.g_loss_sum = tf.summary.scalar("G_loss", self.g_loss)
        self.g_loss_sum_l1 = tf.summary.scalar("G_loss", self.g_loss_l1)
        self.d_loss_sum = tf.summary.scalar("D_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        # print(self.g_vars)
        self.saver = tf.train.Saver()

    def train(self, config):
        g_optim_l1 = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1
                                         ).minimize(self.g_loss_l1, var_list=self.g_vars)

        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1
                                         ).minimize(self.g_loss_total, var_list=self.g_vars) # -self.d_loss_fake
        d_optim = tf.train.AdamOptimizer(0.05*config.learning_rate, beta1=config.beta1
                                                    ).minimize(self.d_loss, var_list=self.d_vars)

        tf.global_variables_initializer().run()

        self.g_sum = tf.summary.merge([self.g_loss_sum, self.d_loss_sum_fake, self.d_fake_sum])
        self.g_sum_l1 = tf.summary.merge([self.g_loss_sum_l1])
        self.d_sum = tf.summary.merge([self.d_loss_sum_real, self.d_real_sum, self.d_loss_sum])
        self.img_sum = tf.summary.merge([self.G_image, self.before_image, self.after_image])
        self.writer = tf.summary.FileWriter(self.writer_path + "/" + self.filename, self.sess.graph)


        train_doublets = self.train_doublets
        train_triplets = self.train_triplets
        val_doublets = self.val_doublets
        val_targets = self.val_targets
        train_singlets = self.train_singlets

        train_triplets_idx = np.arange(train_triplets.shape[0])
        np.random.shuffle(train_triplets_idx)
        # train_doublets_idx = np.arange(train_doublets.shape[0])
        # np.random.shuffle(train_doublets_idx)
        train_doublets_idx = train_triplets_idx


        counter = 1
        start_time = time.time()

        for epoch in range(config.epoch):
            batch_idx = len(train_doublets) // self.batch_size


            for idx in range(0,1): # batch_idx):
                batch_images_idx = train_triplets_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_images = train_triplets[batch_images_idx]

                batch_zs_idx = train_doublets_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_zs = train_doublets[batch_zs_idx]

                batch_targets = train_singlets[batch_images_idx]


                if(config.train_gan):
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={
                                                       self.triplets: batch_images,
                                                       self.doublets: batch_zs,
                                                       self.is_training: True
                                                   })
                    self.writer.add_summary(summary_str, counter)

                    # Update G Network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={
                                                       self.doublets: batch_zs,
                                                       self.is_training: True,
                                                       self.singlets: batch_targets
                                                   })
                    self.writer.add_summary(summary_str, counter)
                else:
                    # Update G Network
                    _, summary_str = self.sess.run([g_optim_l1, self.g_sum_l1],
                                                   feed_dict={
                                                       self.doublets: batch_zs,
                                                       self.is_training: True,
                                                       self.singlets: batch_targets
                                                   })
                    self.writer.add_summary(summary_str, counter)

                counter += 1

                errD_fake = self.d_loss_fake.eval({ self.doublets: batch_zs, self.is_training: True})
                errD_real = self.d_loss_real.eval({ self.triplets: batch_images, self.is_training: True})
                errG = self.g_loss.eval({self.doublets: batch_zs, self.is_training: True})
                errG_l1 = self.g_loss_l1.eval({self.doublets: batch_zs, self.singlets: batch_targets, self.is_training: False})


                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss %.8f, g_loss %.8f, g_loss_l1 %.8f" \
                      % (epoch, idx, batch_idx, time.time() - start_time, errD_fake+errD_real, errG, errG_l1))

            summary_str = self.sess.run(self.img_sum,
                                           feed_dict = {
                                               self.doublets: train_doublets[train_doublets_idx[0:config.batch_size]],
                                               self.is_training: True,
                                               self.mean_placeholder: self.mean_img
                                           })
            self.writer.add_summary(summary_str, counter)


            if np.mod(epoch, 5) == 0:
                self.save(config.checkpoint_dir, counter)

            # Save images to file
            G_img = self.sess.run(self.G + self.mean_img,
                                       feed_dict = {
                                           self.doublets: train_doublets[train_doublets_idx[0:config.batch_size]],
                                           self.is_training: True,
                                           self.mean_placeholder: self.mean_img
                                       })

            G_imgs = [ Image.fromarray(G_img[i], 'RGB') for i in range(G_img.shape[0]) ]
            [ img.save(os.path.join(config.image_dir,"G_epoch%dimg%d.jpeg" %
             (epoch, i))) for i, img in enumerate(G_imgs) ]

            Z_imgs = train_doublets[train_doublets_idx[0:config.batch_size]]
            Z1_imgs = [ Image.fromarray(Z_imgs[i,:,:,:3], 'RGB') for i in range(Z_imgs.shape[0]) ]
            [ img.save(os.path.join(config.image_dir,"Z1_epoch%dimg%d.jpeg" %
             (epoch, i))) for i, img in enumerate(Z1_imgs) ]

            Z2_imgs = [ Image.fromarray(Z_imgs[i,:,:,3:], 'RGB') for i in range(Z_imgs.shape[0]) ]
            [ img.save(os.path.join(config.image_dir,"Z2_epoch%dimg%d.jpeg" %
             (epoch, i))) for i, img in enumerate(Z2_imgs) ]



    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.input_height, self.input_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
