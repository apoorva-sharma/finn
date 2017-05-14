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

def generateDataSet(video_path):
    frames = loadVideoFromPath(video_path)
    downsampled = frames[::2,:,:,:]

    train_befores = downsampled[:-2,:,:,:]
    train_middles = downsampled[1:-1,:,:,:]
    train_afters = downsampled[2:,:,:,:]

    test_befores = downsampled[:-1,:,:,:]
    test_middles = frames[1::2,:,:,:]
    test_afters = downsampled[1:,:,:,:]

    train_doublets = np.concatenate((train_befores, train_afters), axis=3)
    train_triplets = np.concatenate((train_befores, train_middles, train_afters), axis=3)

    test_doublets = np.concatenate((test_befores, test_afters), axis=3)
    test_targets = test_middles

    data = {"train_doublets": train_doublets, "train_triplets": train_triplets,
            "test_doublets": test_doublets, "test_targets": test_targets}

    return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = generateDataSet("datasets/bus_cif.y4m")
    f, axarr = plt.subplots(2, 2)
    axarr[0,0].imshow( 0.5*( data["train_triplets"][0,:,:,0:3] + data["train_triplets"][0,:,:,6:9] ) )
    axarr[1,0].imshow( data["train_triplets"][0,:,:,3:6] )
    axarr[0,1].imshow( 0.5*( data["test_doublets"][0,:,:,0:3] + data["test_doublets"][0,:,:,3:6] ) )
    axarr[1,1].imshow( data["test_targets"][0,:,:,:] )

    #imgplt = plt.imshow(video[5,:,:,:])
    plt.show()
