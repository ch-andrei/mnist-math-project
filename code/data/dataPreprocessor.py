import numpy as np

import tools
import featureExtractor as fe

import cv2

trainXFileNameOut = "processed_train_x.csv"

def preprocess(maxCount=tools.fileLinesCount(fe.trainXFileName), threshold=245):
    with open(fe.trainXFileName, "r") as xFile, \
            open(trainXFileNameOut, "w") as xOutFile:

        count = 0
        for xline in xFile:

            img = np.fromstring(xline, sep=',', dtype=np.uint8)
            img = img.reshape(fe.imgDim)

            imgCopy = np.copy(img)
            imgCopy[imgCopy < threshold] = 0

            # imgComparison = np.zeros((64, 64*2), np.uint8)
            # imgComparison[..., 0 :  64] = img
            # imgComparison[..., 64:2*64] = imgCopy
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
    cv2.waitKey()
