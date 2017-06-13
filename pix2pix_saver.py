import numpy as np
import matplotlib.animation as animation
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import imread
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

    fname = 'football'
    num_frames = 128
    if not interweve:
        frames = []
        for i in range(num_frames):
            if(train):
                frame_G = imread('./' + fname + '_output/%d.jpg' % (i))
                frame_Z2 = imread('./' + fname + '_targets/%d.jpg' % (i))
            else:
                frame_G = imread('./' + fname + '_val_output/images/%d-outputs.png' % (i))
                frame_Z2 = imread('./' + fname + '_val_output/images/%d-targets.png' % (i))


            frames.append( np.concatenate([frame_Z2,frame_G],axis=1) )

        frames = np.array(frames)

        print(frames.shape)

        if(train):
            save_video(frames, './' + fname + '_train.m4v', 15)
        else:
            save_video(frames, './' + fname + '_test.m4v', 15)


    if(interweve):
        video_path = './datasets/' + fname + '_cif.y4m'
        data = generateDataSet(video_path)

        val_doublets = (data["val_doublets"] + np.concatenate([data["mean_img"]]*2,axis=2) )  * 255

        frames = [ val_doublets[0,:,:,0:3] ]
        for i in range(num_frames):
            frame_G = imread('./' + fname + '_val_output/images/%d-outputs.png' % (i))
            frames.append(frame_G)
            frames.append(val_doublets[i,:,:,3:6])

        frames = np.array(frames)
        save_video(frames, './' + fname + '_test_interweve.m4v', 30)



    
    


if __name__ == '__main__':
    main()
