import cv2 as cv
from classification.local_features.local_features import LocalFeatures
from classification.bag_of_words import BagOfWords
from data_io.testdata import TestData
import data_io.settings as Settings


class SIFTClassifier(LocalFeatures):
    """A standard SIFT [60] Classifier using BoW and SVM."""

    

    def __init__(self, testDataObj, svmType=None, name="", description=""):       
            
            
        super(SIFTClassifier, self).__init__(Settings.SI_TESTDATA_SEGMENTS, svmType, Settings.SI_SVM_PARAMS, testDataObj, name, True, description)            
        self.detector = cv.SIFT()
        self.bow = BagOfWords(Settings.SI_BOW_DIMENSION, "SIFT")   
    
    def __getstate__(self):
        result = self.__dict__.copy()
        del result['detector']
        return result 

    def __setstate__(self, dict):
        self.__dict__ = dict
        self.detector = cv.SIFT()

    def __str__(self):
        return "SIFT"
                             









