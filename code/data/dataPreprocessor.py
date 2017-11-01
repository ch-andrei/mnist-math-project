import numpy as np

import tools
import featureExtractor as fe

import cv2

trainXFileNameOut = "processed_train_x.csv"

# consider 3x3 and 9x9 areas, remove areas with less than 4 white pixels
# assumes input where all pixels are either 0 or 255
def removeWhiteSpecs(img):
    imgFloat = img.astype(np.float32) / 255
    imgFloat3 = np.pad(imgFloat, pad_width=1, mode='constant', constant_values=0)
    imgFloat9 = np.pad(imgFloat, pad_width=4, mode='constant', constant_values=0)

    h, w = img.shape[:2]

    kernel = np.ones((9, 9), np.float32)
    convolvedLarge = cv2.filter2D(imgFloat9, -1, kernel)[4:h+4, 4:w+4]

    kernel = np.ones((3, 3), np.float32)
    convolvedSmall = cv2.filter2D(imgFloat3, -1, kernel)[1:h+1, 1:w+1]

    imgCopy = np.copy(img)

    mask = np.zeros((3, h, w), np.uint8)
    mask[0, convolvedSmall <= 2] = 1
    mask[1, convolvedLarge <= 4] = 1
    mask[2] = mask[0] + mask[1]

    imgCopy[mask[2] == 2] = 0

    kernel = np.ones((5, 5), np.float32)
    maskPadded = np.pad(mask[2], pad_width=2, mode='constant', constant_values=0)
    maskPadded[maskPadded > 0] = 1
    maskConvolved = cv2.filter2D(maskPadded, -1, kernel)[2:h+2, 2:w+2]

    mask[2, maskConvolved > 3] = 0
    imgCopy[mask[2] > 0] = 0

    return imgCopy

def preprocess(maxCount=tools.fileLinesCount(fe.trainXFileName), threshold=240):
    with open(fe.trainXFileName, "r") as xFile, \
            open(trainXFileNameOut, "w") as xOutFile:

        count = 0
        for xline in xFile:

            img = np.fromstring(xline, sep=',', dtype=np.uint8)
            img = img.reshape(fe.imgDimTuple)

            imgCopy = np.copy(img)
            imgCopy[imgCopy < threshold] = 0
            imgCopy[imgCopy >= threshold] = 255

            imgNoSpecs = removeWhiteSpecs(imgCopy)

            # for visualization only
            # imgComparison = np.zeros((2 * fe.imgDim, 2 * fe.imgDim), np.uint8)
            # imgComparison[0         :     fe.imgDim, 0         :     fe.imgDim] = img
            # imgComparison[0         :     fe.imgDim, fe.imgDim : 2 * fe.imgDim] = imgCopy
            # imgComparison[fe.imgDim : 2 * fe.imgDim, 0         :     fe.imgDim] = imgCopy - imgNoSpecs
            # imgComparison[fe.imgDim : 2 * fe.imgDim, fe.imgDim : 2 * fe.imgDim] = imgNoSpecs
            # imgComparison = cv2.resize(imgComparison, (0,0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
            # cv2.imshow("img comparison".format(count), imgComparison)
            # cv2.waitKey()

            strX = ""
            imgCopy = imgCopy.flatten()
            for i in range(fe.imgDimSquared):
                strX += str(imgCopy[i]) + ','

            strX = strX[:-1] # remove the last ','
            strX += "\n"

            xOutFile.write(strX)

            count += 1
            if count % 50 == 0:
                print("\rFinished processing {} lines".format(count), end="")

            if count >= maxCount:
                break

if __name__=="__main__":
    preprocess()
