import numpy as np
import matplotlib.animation as animation

def save_video(frames, filename):
    dpi = 100
    img_height = frames.shape[1]
    img_width = frames.shape[2]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(rand(img_height,img_width,3))
    im.set_clim([0,1])
    fig.set_size_inches([img_height/dpi,img_width/dpi])

    tight_layout()

    def update_img(n):
        im.set_data(frames[n,:,:,:].clip(0,1))
        return im

    ani = animation.FuncAnimation(fig, update_img, frames.shape[0], interval=30)
    writer = animation.writers['ffmpeg'](fps=30)

    ani.save(filename,writer=writer,dpi=dpi)
    return ani
