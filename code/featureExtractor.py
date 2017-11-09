import numpy as np
import os, struct, re, cv2, itertools
from os.path import isfile, join
import tools
from keras.datasets import mnist

dataFolderName = "data/"
imgsFolderName = dataFolderName + "imgs/"
originalDataFolderName = dataFolderName + "resources/originalData/"
trainXFileName = "train_x.csv"
trainYFileName = "train_y.csv"
testXFileName = "test_x.csv"
processedTrainXFileName = "processed_train_x.csv"
processedTestXFileName = "processed_test_x.csv"

imgDim = 64
imgDimTuple = (imgDim, imgDim)
imgDimSquared = imgDim * imgDim

dim = 28
dimH = dim
dimW = dim * 4

dimKeras = 224

# dictionary of labels maps value to index
mathValues = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21,
              24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]  # list from Kaggle
mathLabelMap = {value: index for index, value in enumerate(mathValues)}

# because Keras sorts filenames like a potato (unix style)
mathValuesSorted = [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 2, 20, 21, 24, 25, 27, 28, 3, 30,
                    32, 35, 36, 4, 40, 42, 45, 48, 49, 5, 54, 56, 6, 63, 64, 7, 72, 8, 81, 9]
mathLabelMapSorted = {value: index for index, value in enumerate(mathValuesSorted)}

charLabelMap = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'a', 11: 'm'}

def makeFolders(path, folderNames, verbose=False):
    for foldername in folderNames:
        try:
            os.makedirs("{}/{}".format(path, foldername))
        except FileExistsError:
            if verbose:
                print("Folder <{}/{}> already exists.".format(path, foldername))

def readCombinedData(write=False):
    (x_trainM, y_trainM), (x_testM, y_testM) = mnist.load_data()
    (x_trainE, y_trainE), (x_testE, y_testE) = loadEmnistData()

    def emnistGetOnlyAM(xD, yD):
        d = {1:10, 13:11} # 1 is a, 13 is m; thus, the converted labels are: a -> 10, m -> 11
        _x = []
        _y = []
        for i in range(yD.shape[0]):
            y = yD[i]
            if y == 1 or y == 13: # if aA or mM
                _x.append(xD[i])
                _y.append(d[y])
        return _x, _y

    _x_trainE, _y_trainE = emnistGetOnlyAM(x_trainE, y_trainE)
    _x_testE, _y_testE = emnistGetOnlyAM(x_testE, y_testE)

    x_trainE = np.array(_x_trainE)
    y_trainE = np.array(_y_trainE)
    x_testE = np.array(_x_testE)
    y_testE = np.array(_y_testE)

    x_train = np.concatenate((x_trainE, x_trainM), axis=0)
    y_train = np.concatenate((y_trainE, y_trainM), axis=0)
    x_test = np.concatenate((x_testE, x_testM), axis=0)
    y_test = np.concatenate((y_testE, y_testM), axis=0)

    # def resizeTo224(imgs):
    #     count = imgs.shape[0]
    #     dst = np.zeros((count, dimKeras, dimKeras), np.uint8)
    #     for i in range(count):
    #         dst[i] = cv2.resize(imgs[i], (dimKeras, dimKeras), interpolation=cv2.INTER_NEAREST)
    #         if i % 100 == 0:
    #             print("\rfinished {}/{}".format(i, count), end="")
    #     return dst
    #
    # x_train = resizeTo224(x_train)
    # x_test = resizeTo224(x_test)

    def saveToClassifiedFolders(xData, yData, folderName):
        path = "data/chars/" + folderName

        makeFolders(path, charLabelMap.values())

        count = xData.shape[0]
        for i in range(count):
            x, y = xData[i], yData[i]

            if y == 10 or y == 11:
                x = x.transpose() # transpose 'a' and 'm' images

            _folderName = charLabelMap[y]
            imgPath = "{}/{}/img{}.png".format(path, _folderName, i)
            cv2.imwrite(imgPath, x)

            if i % 500 == 0:
                print("\rWrote {}/{} images...".format(i, count), end="")

        print('')

    if write:
        saveToClassifiedFolders(x_train, y_train, "training")
        saveToClassifiedFolders(x_test, y_test, "validation")

    return (x_train, y_train), (x_test, y_test)

def readCustomOld(readCount=None, offset=0, splitRatio=0.25, path ="data/imgs/"):
    filenames = [f for f in os.listdir(path) if isfile(join(path, f)) and f.__contains__(".png")]

    filenames = sorted(filenames, key=lambda filename: int(re.sub("\D", "", filename)))

    if readCount == None:
        pass
    else:
        readCount = min(len(filenames), offset + readCount)

    filenames = filenames[offset: offset + readCount]

    totalFiles = len(filenames)
    data = np.zeros((totalFiles, dimH, dimW), np.uint8)
    for index, filename in enumerate(filenames):

        img = cv2.imread(imgsFolderName + filename, 0)
        data[index] = img

        #
        # for i, permutation in enumerate(permutations):
        #     _img = np.zeros((dim*3, dim), np.uint8)
        #
        #     _img[:dim] = img[permutation[0]:permutation[0]+dim]
        #     _img[dim:dim*2] = img[permutation[1]:permutation[1]+dim]
        #     _img[dim*2:] = img[permutation[2]:permutation[2]+dim]
        #
        #     data[index + i] = _img

        if index % 500 == 0:
            print("\rAcquired {}/{} images...".format(index, totalFiles), end="")
        index += 1

    print("")

    data = np.array(data)

    labels = np.zeros(totalFiles, np.uint8)
    with open(dataFolderName + trainYFileName, encoding='utf-8') as yFile:
        count = 0
        while count < totalFiles:
            line = next(yFile)
            labels[count:count+4] = mathLabelMap[int(line)]
            count += 1

    validCount = int(splitRatio * totalFiles)
    trainCount = totalFiles - validCount

    x_train, y_train = data[:trainCount], labels[:trainCount]
    x_test, y_test = data[trainCount:], labels[trainCount:]

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    return (x_train, y_train), (x_test, y_test)

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

def breakSegmentedIntoThree(format_128x28x1_False_28x28x3_True=True):
    path = "data/imgs/trainImgs/segmented"
    dstPath = "data/imgs/predictTrain/imgs"

    filenames = [f for f in os.listdir(path) if isfile(join(path, f)) and f.__contains__(".png")]
    filenames = sorted(filenames, key=lambda filename: int(re.sub("\D", "", filename)))

    def bbox2range(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    dimO = 28
    eBorder = 2
    dimE = dimO - 2 * eBorder
    for index, filename in enumerate(filenames):
        if format_128x28x1_False_28x28x3_True:
            img = cv2.imread(path + "/" + filename)
        else:
            img = cv2.imread(path + "/" + filename, 0)

        for i in range(3):
            if format_128x28x1_False_28x28x3_True:
                _img = img[..., i]
            else:
                _img = img[:, dimO * (1 + i): dimO * (2 + i)]

            if not _img.sum() == 0:
                rmin, rmax, cmin, cmax = bbox2range(_img)
                rRange, cRange = rmax - rmin, cmax - cmin

                if rRange < 5 or cRange < 5:
                    pass
                else:
                    extracted = _img[rmin:rmax+1, cmin:cmax+1]

                    if rRange > cRange:
                        dimR = int(cRange * dimE / rRange)
                        targetSize = (dimR, dimE)
                        oH, oW = eBorder, int((dimO - dimR) / 2)
                    else:
                        dimR = int(rRange * dimE / cRange)
                        targetSize = (dimE, dimR)
                        oH, oW = int((dimO - dimR) /2), eBorder

                    extracted = cv2.resize(extracted, targetSize, interpolation=cv2.INTER_NEAREST)
                    # TODO: verify interpolation effect

                    _img = np.zeros((dimO, dimO), np.uint8)
                    _img[oH: oH + targetSize[1], oW: oW + targetSize[0]] = extracted
            else:
                print("Found an empty image: ", index)

            cv2.imwrite(dstPath + "/" + "img{}.png".format(str(3 * index + i).zfill(10)), _img)

        if index % 100 == 0:
            print("\rFinished extracting {}/{}".format(index, len(filenames)), end='')

def rawSeparateToFolders():
    path = "data/raw/validationRaw/"

    filenames = [f for f in os.listdir(path) if isfile(join(path, f)) and f.__contains__(".png")]
    filenames = sorted(filenames, key=lambda filename: int(re.sub("\D", "", filename)))

    labelsFilename = path[:-1] + ".csv"
    labels = []
    with open(labelsFilename, encoding='utf-8') as labelsFile:
        for line in labelsFile:
            labels.append(int(line))

    makeFolders(path, mathLabelMap)

    for label, filename in zip(labels, filenames):
        os.rename(path + filename, path + str(label) + "/" + filename)

def renamePadTrainImgsRaw():
    path = "data/imgs/TrainImgs/raw/files/"

    filenames = [f for f in os.listdir(path) if isfile(join(path, f)) and f.__contains__(".png")]
    filenames = sorted(filenames, key=lambda filename: int(re.sub("\D", "", filename)))

    for filename in filenames:
        filenameNew = "img{}.png".format(str(int(re.sub("\D", "", filename))).zfill(10))
        os.rename(path + filename, path + filenameNew)

def loadEmnistData():
    x_train, y_train = readEmnist("training")
    x_test, y_test = readEmnist("testing")
    return (x_train, y_train), (x_test, y_test)

def generateFolderWithClassifiedRawTrainingImages():
    path = "data/resources/rawTrainingData"

    makeFolders(path + "/training", mathValues)
    makeFolders(path + "/validation", mathValues)

    total = 50000
    trainingUntil = 40000

    with open(originalDataFolderName + trainXFileName, 'r') as xFile, \
            open(originalDataFolderName + trainYFileName, 'r') as yFile:

        for index, (xline, yline) in enumerate(zip(xFile, yFile)):

            img = np.fromstring(xline, sep=',').astype(np.uint8).reshape((64, 64))
            value = int(yline.strip())

            if index < trainingUntil:
                cv2.imwrite("{}/training/{}/img{}.png".format(path, value, str(index).zfill(6)), img)
            else:
                cv2.imwrite("{}/validation/{}/img{}.png".format(path, value, str(index).zfill(6)), img)

            if index % 100 == 0:
                print("\rFinished extracting {}/{}".format(index, total), end='')

def generateFolderWithRawTestImages():
    path = "data/resources/rawTestingData/raw"

    with open(originalDataFolderName + testXFileName, 'r') as xFile:

        for index, (xline) in enumerate(xFile):

            img = np.fromstring(xline, sep=',').astype(np.uint8).reshape((64, 64))

            cv2.imwrite("{}/img{}.png".format(path, str(index).zfill(6)), img)

            if index % 100 == 0:
                print("\rFinished extracting {}/{}".format(index, 10000), end='')

def fixOutputs():
    with open('output.csv', 'r') as input, open('_output.csv', 'w') as out:
        for index, line in enumerate(input):
            out.write("{},{}".format(index + 1, line))

def main():
    fixOutputs()
    # pass

if __name__=="__main__":
    main()
