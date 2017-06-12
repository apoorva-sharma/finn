#!/usr/bin/python
import sys

import tensorflow as tf
import numpy as np

# https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value) + 1e-8
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    with tf.variable_scope("ms_ssim_loss"):
        img1 = tf.image.rgb_to_grayscale(img1)
        img2 = tf.image.rgb_to_grayscale(img2)

        weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
        mssim = []
        mcs = []
        for l in range(level):
            ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
            mssim.append(tf.reduce_mean(ssim_map))
            mcs.append(tf.reduce_mean(cs_map))
            filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
            filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
            img1 = filtered_im1
            img2 = filtered_im2

        # list to tensor of dim D+1
        mssim = tf.stack(mssim, axis=0)
        mcs = tf.stack(mcs, axis=0)

        value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                                (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def main():

    import glob
    from scipy.ndimage import imread

    img_dir = sys.argv[1]

    num_images = (len(glob.glob('./'+img_dir+'/*.png')))//3

    inputs = []
    outputs = []
    targets = []

    for i in range(num_images):
        inputs.append(imread('./'+img_dir+'/'+str(i)+'-inputs.png'))
        outputs.append(imread('./'+img_dir+'/'+str(i)+'-outputs.png'))
        targets.append(imread('./'+img_dir+'/'+str(i)+'-targets.png'))

    # inputs = np.array(inputs)
    # outputs = np.array(outputs)
    # targets = np.array(targets)

    H,W,C = inputs[0].shape
    N = len(inputs)
    print('Num images: %d' % N)

    img1 = tf.placeholder(tf.float32, [None, H, W, C], name = 'img1')
    img2 = tf.placeholder(tf.float32, [None, H, W, C], name = 'img2')

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    gan_scores = []
    mssim_scores = []

    with tf.Session(config=run_config) as sess:

        for i in range(N):
            out = sess.run(tf_ms_ssim(img1, img2, mean_metric=False), 
                feed_dict={img1:np.expand_dims(outputs[i],axis=0),img2:np.expand_dims(targets[i],axis=0)})

            gan_scores.append(out)

            out = sess.run(tf_ms_ssim(img1, img2, mean_metric=False), 
                feed_dict={img1:np.expand_dims(inputs[i],axis=0),img2:np.expand_dims(targets[i],axis=0)})

            mssim_scores.append(out)

    mssim_scores = sorted(mssim_scores, reverse=True)
    print( 'MSSIM Generator Score: %.5f' % np.mean(mssim_scores[:50]))

    gan_scores = sorted(gan_scores, reverse=True)
    print( 'MSSIM+GAN Generator Score: %.5f' % np.mean(gan_scores[:50]))







if __name__ == '__main__':
    from datasets import *

    main()
