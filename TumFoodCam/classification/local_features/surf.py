import cv2 as cv
from classification.local_features.local_features import LocalFeatures
from classification.bag_of_words import BagOfWords
from data_io.testdata import TestData
import data_io.settings as Settings


class SURFClassifier(LocalFeatures):
    """ SURF [4] classifier using BoW and SVM/KNN."""
    
    
    def __init__(self, testDataObj, svmType=None, name="", description=""):       
          
        super(SURFClassifier, self).__init__(Settings.SU_TESTDATA_SEGMENTS, svmType, Settings.SU_SVM_PARAMS, testDataObj, name, True, description)            
        self.detector = cv.SURF()
        self.bow = BagOfWords(Settings.SU_BOW_DIMENSION, "SURF")   

    def __getstate__(self):
        result = self.__dict__.copy()
        del result['detector']
        return result 

    def __setstate__(self, dict):
        self.__dict__ = dict
        self.detector = cv.SURF()

    def __str__(self):
        return "SURF"
                             