import numpy as np

import tools
import featureExtractor as fe
from scipy import ndimage

from scipy import stats as st

import data.dataPreprocessor as dp
import cv2

inputFileName = fe.processedTestXFileName
trainXFileNameOut = "segmented_train_x.csv"

doImShow = True
doImWrite = False
imWritePath = "testImgsSegmented128x28/"

# sets pixels far from center to zero based on their Center of Mass
# distanceThreshold is the radius of a unit circle around the center of the image
# all elements whose CoM
def suppressEdgeElements(img, distanceThreshold=0.55):
    h, w = img.shape[:2]
    maxRtoKeep = (h + w) / 4 * distanceThreshold
    imageCenter = np.array([h / 2, w / 2], np.float32)

    distanceMask = np.zeros(img.shape, np.float32)

    _, markers = cv2.connectedComponents(img)

    numElements = markers.max()
    for i in range(numElements):
        eId = i + 1
        ePixels = np.where(markers == eId)

        pixelCount = ePixels[0].shape[0]
        ePixX = ePixels[0].sum() # average x coord of all pixels in the current group
        ePixY = ePixels[1].sum() # average y coord of all pixels in the current group

        eCenter = np.array([ePixX / pixelCount, ePixY / pixelCount], np.float32)

        distanceMask[ePixels] = np.linalg.norm(eCenter - imageCenter)

    imgCopy = np.copy(img)
    imgCopy[distanceMask > maxRtoKeep] = 0

    return imgCopy

def processImages(maxCount=None, extractRadius=20):
    if maxCount == None:
        maxCount = tools.fileLinesCount(inputFileName)

    extractRadius2 = extractRadius * 2

    with open(inputFileName, "r") as xFile:

        count = 0

        # extract characters as subimages using image's moments as character centers
        toSkip = 0

        skipCount = 0
        for xline in xFile:
            try:
                if skipCount < toSkip:
                    print("skipping!")
                    skipCount += 1
                    count += 1
                    continue

                # create local vars so these can be modified on errors; init to global vars
                doImshow = doImShow
                doImwrite = doImWrite

                img = np.fromstring(xline, sep=',', dtype=np.uint8)
                img = img.reshape(fe.imgDimTuple)

                # compute image contours
                _, contours, _ = cv2.findContours(np.copy(img), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

                # compute image moment centers
                centers = []
                for i in range(len(contours)):
                    moments = cv2.moments(contours[i])
                    center = (int(moments['m10'] / (moments['m00'] + 0.0001)),
                              int(moments['m01'] / (moments['m00'] + 0.0001))) # 0.0001 to avoid dividing by 0
                    centers.append(center)

                # remove extrema centers (on the edges)
                offset=0
                for i in range(len(centers)):
                    p1 = centers[i - offset]
                    if p1[0] == 0 or p1[1] == 0 or p1[0] == fe.imgDim or p1[1] == fe.imgDim:
                        centers.remove(p1)
                        offset += 1

                # remove diplicates
                centers = list(set(centers))

                # sort to ascending order
                centers = sorted(centers)

                # apply kmeans if more than 3 centers
                if len(centers) > 3:
                    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    flags = cv2.KMEANS_RANDOM_CENTERS

                    centers = np.array(centers, np.float32) # list of tuples to numpy array

                    attempts = 5
                    compactness, labels, centers = cv2.kmeans(centers, 3, None, criteria, attempts, flags)

                # widen distance between centers if these are too close
                repulsionStrength = 0.75
                repulseDistance = extractRadius2 * 3 / 4
                minDistanceToCombine = extractRadius / 4
                maxRepulseMagnitude = extractRadius

                _centers = np.array(centers, np.float32)
                centersBeforeRepulse = np.copy(_centers)

                n = _centers.shape[0]
                p = np.where(np.tril(np.ones((n, n)), -1) > 0)
                # combine or repulse
                for i, j in zip(p[0], p[1]):
                    # the two compared centers
                    c = _centers[i]
                    _c = _centers[j]

                    # difference vector
                    v = c - _c
                    vM = np.linalg.norm(v) # magnitude
                    vN = v / vM # normalized direction

                    # are centers close enough?
                    if vM < repulseDistance:

                        # repulse or combine
                        if vM > minDistanceToCombine:
                            # repulse centers
                            repulseMagnitude = repulseDistance - vM

                            if repulseMagnitude > maxRepulseMagnitude or np.isnan(repulseMagnitude):
                                repulseMagnitude = maxRepulseMagnitude

                            repulseVector = repulsionStrength * repulseMagnitude * vN

                            if np.isnan(repulseVector.any()):
                                continue

                            _centers[i] = c + repulseVector
                            _centers[j] = _c - repulseVector
                        else:
                            # combine centers instead of repulsing

                            # average
                            avgC = (c + _c) / 2

                            c = avgC
                            _c = avgC

                centers = np.round(_centers).astype(np.uint8)

                # remove duplicates
                centers = np.unique(centers, axis=0)
                centers = centers.tolist()

                # make sure at least 3 centers are available
                while len(centers) < 3:
                    # TODO: improve this to generate better new centers
                    centers.append(centers[0])

                centers = np.array(centers)

                centersImg = np.zeros((fe.imgDim, fe.imgDim, 3), np.uint8)
                extractedCenters = np.zeros((extractRadius2, extractRadius2, 3), np.uint8)
                for i in range(len(centers)):
                    p1 = centers[i].astype(np.int8) - extractRadius
                    p2 = centers[i].astype(np.int8) + extractRadius

                    # clamp edges
                    p1[p1 < 0] = 0
                    p2[p2 < 0] = 0
                    p1[p1 > fe.imgDim - 1] = fe.imgDim - 1
                    p2[p2 > fe.imgDim - 1] = fe.imgDim - 1

                    pInd = p2 - p1 # size of the section to extract
                    pIndOffset = extractRadius2 - pInd # center offset

                    # get the raw subimage
                    extracted = np.zeros((extractRadius2, extractRadius2))
                    extracted[pIndOffset[1]: pInd[1] + pIndOffset[1],
                                     pIndOffset[0]: pInd[0] + pIndOffset[0]] \
                        = img[p1[1]: p2[1],
                              p1[0]: p2[0]]

                    # adjust the subimage in such a way that the extracted character is as centered as possible
                    # compute center of mass (CoM) and then a vector for CoM displacement to centralize CoM
                    centerOfMass = np.array(ndimage.measurements.center_of_mass(extracted)).transpose()
                    centerDisplaceVector = (extractRadius - np.round(centerOfMass)).astype(np.int8)

                    # limit max CoM displacement
                    centerDisplaceVector = np.clip(centerDisplaceVector, -int(extractRadius / 4), int(extractRadius / 4))

                    # shift CoM
                    shifted = np.zeros((extractRadius2, extractRadius2), np.uint8)
                    shifted[max(0, centerDisplaceVector[0]): extractRadius2 + centerDisplaceVector[0],
                            max(0, centerDisplaceVector[1]): extractRadius2 + centerDisplaceVector[1]] = \
                        extracted[
                            max(0, -centerDisplaceVector[0]): extractRadius2 - centerDisplaceVector[0],
                            max(0, -centerDisplaceVector[1]): extractRadius2 - centerDisplaceVector[1]]

                    # remove elements near the edges # important to do so after shifting
                    suppressed = suppressEdgeElements(shifted.astype(np.uint8))

                    # finally, copy the extracted sample
                    extractedCenters[..., i] = suppressed

                    # just for visualisation
                    # mark the extracted area as a red rectangle over the original image
                    blank = np.zeros(fe.imgDimTuple, np.uint8)
                    cv2.rectangle(blank, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 85, -1)
                    centersImg[..., 2] += blank # for a semi transparent effect

                # debugging before repulse
                for i in range(len(centersBeforeRepulse)):
                    p1 = centersBeforeRepulse[i].astype(np.int8) - extractRadius
                    p2 = centersBeforeRepulse[i].astype(np.int8) + extractRadius

                    # clamp edges
                    p1[p1 < 0] = 0
                    p2[p2 < 0] = 0
                    p1[p1 > fe.imgDim - 1] = fe.imgDim - 1
                    p2[p2 > fe.imgDim - 1] = fe.imgDim - 1

                    # just for visualisation
                    # mark the extracted area as a red rectangle over the original image
                    blank = np.zeros(fe.imgDimTuple, np.uint8)
                    cv2.rectangle(blank, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 35, -1)
                    centersImg[..., 1] += blank # for a semi transparent effect

                if doImshow:
                    # just for visualization
                    centersImg[..., 0] = img # put the original image in the blue channel
                    centersImg = tools.resize(centersImg, 10, 10)
                    _extractedCenters = tools.resize(extractedCenters)
                    cv2.imshow("centersImg".format(count), centersImg)
                    cv2.imshow("extractedCenters1", _extractedCenters)
                    # cv2.imshow("extractedCenters1", _extractedCenters[..., 0])
                    # cv2.imshow("extractedCenters2", _extractedCenters[..., 1])
                    # cv2.imshow("extractedCenters3", _extractedCenters[..., 2])
                    cv2.waitKey(0)

                if doImwrite:
                    cv2.imwrite(imWritePath + "img{}.png".format(count), extractedCenters)

                # TODO: write extractedCenters[1,2,3] to file
                # for i in range(3):
                #     for row in extractedCenters[..., i]:
                #         print("row " + str(row))

            except cv2.error as e:
                print("cv2 error: {}".format(e))
                pass

            count += 1
            if count % 50 == 0:
                print("\rFinished processing {}/{} lines".format(count, maxCount - toSkip), end="")

            if count >= maxCount:
                break

if __name__=="__main__":
    processImages()
