import numpy as np

import tools
import featureExtractor as fe

outputFileNameOut = "_test_x.csv"

def convertToEfficientFormat(maxCount=tools.fileLinesCount(fe.testXFileName)):
    with open(fe.testXFileName, "r") as xFile, \
            open(outputFileNameOut, "w") as xOutFile:

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
                print("\rFinished processing {}/{} lines".format(count, maxCount), end="")

            if count >= maxCount:
                break

if __name__=="__main__":
    convertToEfficientFormat()
