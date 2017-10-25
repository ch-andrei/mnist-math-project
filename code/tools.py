import os
import pickle as pk

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
