import numpy as np

import tools
import featureExtractor as fe

trainXFileNameOut = "_train_x.csv"

def convertToEfficientFormat(maxCount=tools.fileLinesCount(fe.trainXFileName)):
    with open(fe.trainXFileName, "r") as xFile, \
            open(trainXFileNameOut, "w") as xOutFile:

        count = 0
        for xline in xFile:
            strX = ""

            xs = xline.split(',')
            for i in range(len(xs)):
                strX += str(int(float(xs[i]))) + ','

            strX = strX[:-1] # remove the last ','
            strX += "\n"

            xOutFile.write(strX)

            count += 1

            if count % 50 == 0:
                print("\rFinished processing {} lines".format(count), end="")

            if count >= maxCount:
                break

convertToEfficientFormat()
