import cv2 as cv
import numpy as np
from classification.local_features.local_features import LocalFeatures
from classification.bag_of_words import BagOfWords
from data_io.testdata import TestData
import data_io.settings as Settings


class ORBClassifier(LocalFeatures):
    """A standard ORB [74] classifier using BoW and SVM/KNN."""
    
    def __init__(self, testDataObj, svmType=None, testData="", name="", description=""):       
       
        super(ORBClassifier, self).__init__(Settings.O_TESTDATA_SEGMENTS, svmType, Settings.O_SVM_PARAMS, testDataObj, name, True, description)            
        self.detector = cv.ORB()
        self.bow = BagOfWords(Settings.O_BOW_DIMENSION, dextractor="ORB", dmatcher="BruteForce", vocabularyType=np.uint8)   

    def __getstate__(self):
        result = self.__dict__.copy()
        del result['detector']
        return result 

    def __setstate__(self, dict):
        self.__dict__ = dict
        self.__detector = cv.ORB()

    def __str__(self):
        return "ORB"
