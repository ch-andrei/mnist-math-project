import threading

import cv2
import numpy as np
import itertools

from keras.preprocessing.image import ImageDataGenerator

import featureExtractor as fe
import tools

# debug = True
debug = False

# data augmentation generator
class TrioDatasetGenerator():
    """
    this class follows the interface of Keras image generator and can be used with model.fit_generator(...).
    At each batch of size N, this generator will return N images containing three characters (2 digits and 1 letter),
    and the associated labels containing the expected mathematical outputs (in 1-hot encoding form).
    """
    def __init__(self, path,
                 imgDim=64,
                 channels=1,
                 batchSize=32,
                 rotation_range=45,
                 shearRange=0.1,
                 zoomRange=0.1,
                 rescale=1.0/255.0,
                 minDistBetweenCenters=0.55,
                 doPixelSuppress=True,
                 suppressPixelsThreshold=200,
                 injectNoise=False,
                 noiseAmount=0.1,
                 getMathOutputs=False  # when false, generate center coords as labels; when true, math outputs
                 ):
        self.imgDim = imgDim
        self.channels = channels

        self.rescale = rescale
        self.batchSize = batchSize
        self.numClasses = 40 if getMathOutputs else 6 # 40 possible math outputs; 3 centers with 2 coords, hence 6
        self.getMathOutputs = getMathOutputs

        # characters dimensions
        self.charDim = 28
        self.input_shape = (self.charDim, self.charDim)

        # matte dim for characters
        self.imgCharDim = 64
        self.coordRange = self.imgCharDim - self.charDim
        self.coordRescale = self.coordRange / self.imgCharDim
        self.coordOffset = self.charDim / self.imgCharDim / 2

        self.minDistBetweenCenters = minDistBetweenCenters

        self.doPixelSuppress = doPixelSuppress
        self.suppressPixelsThreshold = suppressPixelsThreshold

        self.injectNoise = injectNoise # TODO

        self.train_datagen_digits = ImageDataGenerator(
            rotation_range=rotation_range,  # up to 45 degrees random rotation
            shear_range=shearRange,
            zoom_range=zoomRange,
            rescale=None
        )

        self.train_generator_digits = self.train_datagen_digits.flow_from_directory(
            path + "/digits",
            color_mode='grayscale',
            target_size=self.input_shape,
            batch_size=self.batchSize,
            shuffle=True # IMPORTANT: shuffle must be true
        )

        self.train_datagen_letters = ImageDataGenerator(
            rotation_range=rotation_range,  # up to 45 degrees random rotation
            shear_range=shearRange,
            zoom_range=zoomRange,
            rescale=None
        )

        self.train_generator_letters = self.train_datagen_letters.flow_from_directory(
            path + "/letters",
            color_mode='grayscale',
            target_size=self.input_shape,
            batch_size=self.batchSize,
            shuffle=True # IMPORTANT: shuffle must be true
        )

    def __iter__(self):
        return self

    # returns the top left coordinate of a square
    def nextRandomSquares(self):
        centers = []

        while len(centers) < 3:
            center = np.random.rand(2)
            append = True
            for _center in centers:
                if np.linalg.norm(_center-center) < self.minDistBetweenCenters:
                    append = False
            if append:
                centers.append(center)

        centers = sorted(centers, key=lambda x: x[0])
        centers = np.array(centers, np.float32)

        return np.clip(centers.flatten() * self.coordRescale + self.coordOffset, 0, 1), \
               (centers * self.coordRange).astype(np.uint8)  # (tuple: actual centers as [0-1], top left indices)

    def __next__(self):
        # get batches of data
        digit1Batch = next(self.train_generator_digits)
        digit2Batch = next(self.train_generator_digits)
        letterBatch = next(self.train_generator_letters)

        maxBatchSize = min(digit1Batch[0].shape[0], digit2Batch[0].shape[0], letterBatch[0].shape[0])

        imgs = np.zeros((maxBatchSize, self.imgDim, self.imgDim, self.channels), np.float32)

        labels = np.zeros((maxBatchSize, self.numClasses), np.float32)

        if self.getMathOutputs:
            output = np.zeros((maxBatchSize, 40), np.uint8)
        else:
            output = np.zeros((maxBatchSize, 3, 12), np.float32)

        for i in range(maxBatchSize):

            digit1 = digit1Batch[0][i] # characters
            digit1Label = digit1Batch[1][i] # labels

            digit2 = digit2Batch[0][i] # characters
            digit2Label = digit2Batch[1][i] # labels

            letter = letterBatch[0][i]
            letterLabel = letterBatch[1][i]

            _centers, centers = self.nextRandomSquares()

            img = np.zeros((64, 64, self.channels), np.float32)

            permutation = np.random.permutation(3)
            centers = np.array([centers[i] for i in permutation])

            img[centers[0, 0]: centers[0, 0] + self.charDim, centers[0, 1]: centers[0, 1] + self.charDim] += digit1
            img[centers[1, 0]: centers[1, 0] + self.charDim, centers[1, 1]: centers[1, 1] + self.charDim] += digit2
            img[centers[2, 0]: centers[2, 0] + self.charDim, centers[2, 1]: centers[2, 1] + self.charDim] += letter

            img = img.clip(0, 255.0) # handle excessive values

            # resize if needed
            if self.imgCharDim != self.imgDim:
                img = cv2.resize(img, (self.imgDim, self.imgDim), interpolation=cv2.INTER_NEAREST)
                img = img.reshape((self.imgDim, self.imgDim, self.channels))

            if self.doPixelSuppress:
                img = self.suppressPixelsImg(img)

            # rescale image values (ex: from [0-255] to [0-1] range, when self.rescale=1./255.)
            img = img * self.rescale

            if self.injectNoise:
                img = self.injectNoise(img)

            # finally, copy image to output array
            imgs[i] = img

            if self.getMathOutputs:
                # compute the expected mathematical result
                op1 = np.argmax(digit1Label)
                op2 = np.argmax(digit2Label)
                op = np.argmax(letterLabel)

                if op == 0:
                    value = op1 + op2
                else:
                    value = op1 * op2

                output[i, fe.mathLabelMapSorted[value]] = 1
            else:
                # labels of included characters
                output[i, permutation[0], :10] = digit1Label
                output[i, permutation[1], :10] = digit2Label
                output[i, permutation[2], 10:] = letterLabel

                # center coordinates
                labels[i] = _centers

            if debug:
                # show generate image and print some stats

                dim = self.imgDim
                _img = np.zeros((dim, dim, 3))

                _img[..., 2] = imgs[i, 0].reshape((dim,dim))
                _centers = _centers.reshape((3,2))

                _img[int(_centers[0, 0] * dim), int(_centers[0, 1] * dim), 0] = 1
                _img[int(_centers[1, 0] * dim), int(_centers[1, 1] * dim), 0] = 1
                _img[int(_centers[2, 0] * dim), int(_centers[2, 1] * dim), 0] = 1

                print(labels[i], output[i])
                cv2.imshow("img", tools.resize(_img, 2, 2))
                cv2.waitKey()

        if self.getMathOutputs:
            return (imgs, output)
        else:
            return (imgs, labels)

    def next(self):
        return self.__next__()

    def suppressPixelsImg(self, img):
        img[img < self.suppressPixelsThreshold] = 0
        return img

    def injectNoiseToImg(self, img):
        # TODO: finish this
        return img


if __name__=="__main__":
    trioGen = TrioDatasetGenerator("resources/charsDigitsLetters/training/", imgDim=224, channels=3, getMathOutputs=True)
    trioGen.next()
