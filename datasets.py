import y4m
import numpy as np

YCbCr2RGB = np.array( [[1.164, 0., 1.793],
                       [1.164, -0.213, -0.533],
                       [1.164, 2.112, 0.0]] )

def frameToNDArray(frame):
    data = np.frombuffer(frame.buffer, dtype='uint8').astype('float32')
    H = frame.headers['H']
    W = frame.headers['W']
    data = np.reshape(data, [6, H//2, W//2])
    Y = data[0:4,:,:].reshape([H, W])
    Cb = data[4,:,:].clip(16, 240)-128
    Cb = Cb.repeat(2, axis=0).repeat(2, axis=1)
    Cr = data[5,:,:].clip(16, 240)-128
    Cr = Cr.repeat(2, axis=0).repeat(2, axis=1)
    #import pdb; pdb.set_trace()
    frame = np.stack([Y, Cb, Cr], axis=-1)
    frameRGB = np.dot(frame, YCbCr2RGB.T)
    return frameRGB.clip(0,255)/255

class Y4MDecoder:
    def __init__(self):
        self.frames = []

    def incorporateFrame(self,frame):
        self.frames.append(frameToNDArray(frame))

    def getVideo(self):
        return np.stack(self.frames)


def loadVideoFromPath(path):
    decoder = Y4MDecoder()
    parser = y4m.Reader(decoder.incorporateFrame, verbose=False)
    with open(path, 'r') as f:
        parser.decode(f.read())

    return decoder.getVideo()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    video = loadVideoFromPath("datasets/bus_cif.y4m")
    imgplt = plt.imshow(video[5,:,:,:])
    plt.show()
