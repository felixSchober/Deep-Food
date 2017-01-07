import sys
import cPickle
import os
import errno
import cv2 as cv
import numpy as np
import math
from math import ceil, sqrt
import datetime
import sys

# simple module to convert the CIFAR source files to the needed dataset format.

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def create_dir_if_necessary(path):
    """ Save way for creating a dir if necessary. 
    From http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def show_progress(show, current, max, text, *args):
    if show:
        progress = round((float(current) / max) * 100.0, 0)
        output = "\r" + text.format(*args) + " {0}% done.       ".format(progress)                    
        sys.stdout.write(output)
        sys.stdout.flush() 

def convert(inputPath, outputPath, batch):
    print "\n\nOpening",inputPath + str(batch)
    dataDict = unpickle(inputPath + str(batch))
    x = dataDict["data"]
    x = x.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype('float32')
    y = dataDict["labels"]
    size = len(x)
    for i in xrange(size):
        img = x[i]
        lbl = y[i]
        create_dir_if_necessary(outputPath + str(lbl))
        cv.imwrite(outputPath + str(lbl) + "/" + str(i) + batch + ".jpg", img)
        show_progress(True, i, size, "Progress:")
    print "finished"


path = "/home/ubuntu/Datasets/cifar-10-batches-py/"
output = "/home/ubuntu/Datasets/cifar-folders/"
convert(path, output, "data_batch_1")
convert(path, output, "data_batch_2")
convert(path, output, "data_batch_3")
convert(path, output, "data_batch_4")
convert(path, output, "data_batch_5")
