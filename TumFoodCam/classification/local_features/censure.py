from classification.local_features.local_features import LocalFeatures
from classification.bag_of_words import BagOfWords
from data_io.testdata import TestData
import data_io.settings as Settings
from skimage.feature import CENSURE
from misc import utils
import cv2 as cv


class CENSUREClassifier(LocalFeatures):
    """
    CenSurE feature detector [2]. SURF [4, 5] as feature descriptor.
    Censure implementation from skimage
    """


    def __init__(self, testDataObj, svmType=None, name="", description=""):         
        super(CENSUREClassifier, self).__init__(Settings.C_TESTDATA_SEGMENTS, svmType, Settings.C_SVM_PARAMS, testDataObj, name, True, description)            
        self.detector = None
        self.bow = BagOfWords(Settings.C_BOW_DIMENSION, "SURF")    
        self.descriptor = cv.SURF()
        self.detector = CENSURE(non_max_threshold=0.01)

    def detect_keypoints(self, image):
        self.detector.detect(image)
        keypointList = self.detector.keypoints

        # convert coordinate keypoints to opencv keypoints
        # regarding class_id and size values see http://stackoverflow.com/questions/17981126/what-is-the-meaning-and-use-of-class-member-class-id-of-class-cvkeypoint-in-op and http://stackoverflow.com/questions/34104297/how-to-convert-given-coordinates-to-kaze-keypoints-in-python-with-opencv 
        return [cv.KeyPoint(p[0], p[1], 5, _class_id=0) for p in keypointList]

    def descibe_keypoints(self, image, keypoints):
        return self.descriptor.compute(image, keypoints)




    def __getstate__(self):
        result = self.__dict__.copy()
        del result['detector']
        return result 

    def __setstate__(self, dict):
        self.__dict__ = dict
        self.detector = cv.SURF()

    def __str__(self):
        return "CENSURE"


