import numpy as np
import os, struct

import tools

dataFolderName = "data/"
trainXFileName = "train_x.csv"
trainYFileName = "train_y.csv"
processedTrainXFileName = "processed_train_x.csv"

imgDim = 64
imgDimTuple = (imgDim, imgDim)
imgDimSquared = imgDim * imgDim

def readEmnist(dataset ="training", path ="data/emnist/"):

    if dataset is "training":
        fname_img = os.path.join(path, 'emnist-letters-train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'emnist-letters-train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 'emnist-letters-test-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'emnist-letters-test-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    print(lbl.shape, img.shape)

    return img, lbl

def loadTrainTestData():
    x_train, y_train = readEmnist("training")
    x_test, y_test = readEmnist("testing")
    return (x_train, y_train), (x_test, y_test)

def main():
    pass

if __name__=="__main__":
    main()
