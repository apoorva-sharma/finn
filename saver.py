import numpy as np
import matplotlib.animation as animation
import glob
from scipy.misc import imread
import matplotlib.pyplot as plt
from pylab import *
from datasets import *


def save_video(frames, filename, fps):
    dpi = 100
    img_height = frames.shape[1]
    img_width = frames.shape[2]

    print(img_height, img_width)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(rand(img_height,img_width,3))
    im.set_clim([0,1])
    fig.set_size_inches([img_width/dpi,img_height/dpi])

    tight_layout()

    def update_img(n):
        im.set_data(frames[n,:,:,:].clip(0,1))
        return im

    ani = animation.FuncAnimation(fig, update_img, frames.shape[0], interval=fps)
    writer = animation.writers['ffmpeg'](fps=fps, bitrate=8*512)

    ani.save(filename,writer=writer,dpi=dpi)
    return ani


def main():

    train = False

    interweve = False


    if not interweve:
        frames = []
        for i in range(148):
            if(train):
                frame_G = imread('./news_images/G_epoch95img%d.jpeg' % (i))
                frame_Z2 = imread('./news_images/Z2_epoch0img%d.jpeg' % (i))
                frame_Z13 = imread('./news_images/Z13_epoch0img%d.jpeg' % (i))
            else:
                frame_G = imread('./news_images/G_valimg%d.jpeg' % (i))
                frame_Z2 = imread('./news_images/Z2_valimg%d.jpeg' % (i))
                frame_Z13 = imread('./news_images/Z13_valimg%d.jpeg' % (i))


            frames.append( np.concatenate([frame_Z2,frame_G,frame_Z13],axis=1) )

        frames = np.array(frames)

        print(frames.shape)

        if(train):
            save_video(frames/255, './news_train.m4v', 15)
        else:
            save_video(frames/255, './news_test.m4v', 15)


    if(interweve):
        video_path = './datasets/football_cif.y4m'
        data = generateDataSet(video_path)

        val_doublets = (data["val_doublets"] + np.concatenate([data["mean_img"]]*2,axis=2) )  * 255

        frames = [ val_doublets[0,:,:,0:3] ]
        for i in range(128):
            frame_G = imread('./football_images/G_valimg%d.jpeg' % (i))
            frames.append(frame_G)
            frames.append(val_doublets[i,:,:,3:6])

        frames = np.array(frames)
        save_video(frames/255, './football_test_interweve.m4v', 30)



    
    


if __name__ == '__main__':
    main()