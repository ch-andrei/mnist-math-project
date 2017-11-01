import os, time
import pickle as pk
import numpy as np
import scipy.stats as st
import cv2

def resize(img, fx=5, fy=5):
    return cv2.resize(img, (0,0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)

def checkForExistingDataFile(path):
    return os.path.isfile(path)

def pickleReadOrWriteObject(path, object=None):
    if object == None:
        if checkForExistingDataFile(path):
            with open(path, "rb") as f:
                print("Reading pickle [{}]...".format(path))
                return pk.load(f)
        return None
    else:
        with open(path, "wb") as f:
            print("Dumping to pickle [{}]...".format(path))
            pk.dump(object, f)

def fileLinesCount(fname):
    with open(fname, "r", encoding="utf8") as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def gkern(kernlen=21, nsig=2):
    """Returns a 2D Gaussian kernel array."""

    interval = (2 * nsig + 1.0) / (kernlen)
    x = np.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

# usage:
# with Timer("Timer's name):
#     [...code...]
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s] ' % self.name)
        print('Elapsed: %.2f seconds' % (time.time() - self.tstart))
