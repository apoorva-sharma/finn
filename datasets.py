import y4m
import numpy as np
import codecs

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
    with open(path, 'rb') as f:
        data = f.read()
        parser.decode(data.decode('latin-1', errors='replace').encode('latin-1'))

    return decoder.getVideo()

def generateDataSet(video_path):
    frames = loadVideoFromPath(video_path)
    downsampled = frames[::2,:,:,:]

    mean_img = np.mean(downsampled, axis=0)
    restore = lambda img: img + mean_img

    train_befores = downsampled[:-2,:,:,:] - mean_img
    train_middles = downsampled[1:-1,:,:,:] - mean_img
    train_afters = downsampled[2:,:,:,:] - mean_img

    val_befores = downsampled[:-1,:,:,:] - mean_img
    val_middles = frames[1::2,:,:,:] - mean_img
    val_afters = downsampled[1:,:,:,:] - mean_img

    train_doublets = np.concatenate((train_befores, train_afters), axis=3)
    train_triplets = np.concatenate((train_befores, train_middles, train_afters), axis=3)

    val_doublets = np.concatenate((val_befores, val_afters), axis=3)
    val_targets = val_middles

    data = {"train_doublets": train_doublets, "train_triplets": train_triplets,
            "val_doublets": val_doublets, "val_targets": val_targets,
            "mean_img": mean_img, "restore": restore}

    return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = generateDataSet("datasets/bus_cif.y4m")
    f, axarr = plt.subplots(2, 2)
    axarr[0,0].imshow( data["restore"]( 0.5*( data["train_triplets"][0,:,:,0:3] + data["train_triplets"][0,:,:,6:9] ) ) )
    axarr[1,0].imshow( data["restore"]( data["train_triplets"][0,:,:,3:6] ) )
    axarr[0,1].imshow( data["restore"]( 0.5*( data["val_doublets"][0,:,:,0:3] + data["val_doublets"][0,:,:,3:6] ) ) )
    axarr[1,1].imshow( data["restore"]( data["val_targets"][0,:,:,:] ) )

    #imgplt = plt.imshow(video[5,:,:,:])
    plt.show()
