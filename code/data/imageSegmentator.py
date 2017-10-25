import numpy as np

import tools
import featureExtractor as fe
from scipy import ndimage

from scipy import stats as st

import cv2

trainXFileNameOut = "processed_train_x.csv"

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def preprocess(maxCount=1000, extractRadius=20): #tools.fileLinesCount(fe.trainXFileName)):
    extractRadius2 = extractRadius * 2
    with open(fe.processedTrainXFileName, "r") as xFile:
        count = 0
        for xline in xFile:
            try:
                img = np.fromstring(xline, sep=',', dtype=np.uint8)
                img = img.reshape(fe.imgDim)

                _, contours, _ = cv2.findContours(np.copy(img), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

                centers = []
                for i in range(len(contours)):
                    moments = cv2.moments(contours[i])
                    centers.append((int(moments['m10'] / (moments['m00'] + 0.0001)), int(moments['m01'] / (moments['m00'] + 0.0001))))

                # remove moments that are on the edges
                offset=0
                for i in range(len(centers)):
                    p1 = centers[i - offset]
                    if p1[0] == 0 or \
                        p1[0] == 64 or \
                        p1[1] == 0 or \
                        p1[1] == 64:
                        centers.remove(p1)
                        offset += 1

                # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                flags = cv2.KMEANS_RANDOM_CENTERS

                # Apply KMeans
                centers = np.array(centers, np.float32)

                compactness, labels, centers = cv2.kmeans(centers, 3, None, criteria, 10, flags)

                weightKernel = gkern(extractRadius2)[:extractRadius2, :extractRadius2]
                weightKernel /= weightKernel.max()

                centersImg = np.zeros((64, 64, 3), np.uint8)
                extractedCenters = np.zeros((extractRadius2, extractRadius2, 3), np.uint8)
                for i in range(len(centers)):
                    p1 = centers[i].astype(np.int8) - extractRadius
                    p2 = centers[i].astype(np.int8) + extractRadius

                    p1[p1 < 0] = 0
                    p2[p2 < 0] = 0
                    p1[p1 > 63] = 63
                    p2[p2 > 63] = 63

                    pInd = p2 - p1
                    pIndOffset = extractRadius2 - pInd # center offset

                    extracted = np.zeros((extractRadius2, extractRadius2))

                    extracted[pIndOffset[1]: pInd[1] + pIndOffset[1],
                                     pIndOffset[0]: pInd[0] + pIndOffset[0]] \
                        = img[p1[1] : p2[1],
                              p1[0] : p2[0]]

                    # scale edges to weigh less
                    weighted = (extracted * weightKernel).astype(np.uint8)
                    centerOfMass = np.array(ndimage.measurements.center_of_mass(weighted)).transpose()
                    centerDisplaceVector = (extractRadius - np.round(centerOfMass)).astype(np.int8)

                    print(centerDisplaceVector)

                    shifted = np.zeros((extractRadius2, extractRadius2))
                    shifted[max(0, centerDisplaceVector[0]) : extractRadius2 + centerDisplaceVector[0],
                            max(0, centerDisplaceVector[1]) : extractRadius2 + centerDisplaceVector[1]] = \
                                extracted[
                                    max(0, -centerDisplaceVector[0]) : extractRadius2 - centerDisplaceVector[0],
                                    max(0, -centerDisplaceVector[1]) : extractRadius2 - centerDisplaceVector[1]]

                    extractedCenters[..., i] = shifted

                    # just for visualisation
                    blank = np.zeros((64, 64, 3), np.uint8)
                    cv2.rectangle(blank, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,0,50), -1)

                    centersImg += blank

                centersImg[..., 0] = img

                centersImg = cv2.resize(centersImg, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
                extractedCenters = cv2.resize(extractedCenters, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
                cv2.imshow("centersImg".format(count), centersImg)
                cv2.imshow("extractedCEnters", extractedCenters)
                cv2.waitKey()

            except cv2.error as e:
                print("EXCEPTION!")
                pass

            count += 1
            if count % 50 == 0:
                print("\rFinished processing {} lines".format(count), end="")

            if count >= maxCount:
                break

if __name__=="__main__":
    preprocess()
