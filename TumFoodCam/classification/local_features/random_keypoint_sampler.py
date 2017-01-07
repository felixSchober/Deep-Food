from classification.local_features.local_features import LocalFeatures
from classification.bag_of_words import BagOfWords
from data_io.testdata import TestData
import data_io.settings as Settings
import numpy as np
import cv2 as cv


class RandomKeypointSampler(LocalFeatures):
    """Random keypoint sampling around the image center."""

    def __init__(self, testDataObj, svmType=None, name="", description=""):           
        super(RandomKeypointSampler, self).__init__(Settings.R_TESTDATA_SEGMENTS, svmType, Settings.R_SVM_PARAMS, testDataObj, name, True, description)            
        self.bow = BagOfWords(Settings.R_BOW_DIMENSION, "SURF")    
        self.detector = cv.SURF()

    def detect_keypoints(self, image):
        """
        This method overrides the detect_keypoints method from EdgeClassifier.

        """
        width = image.shape[1]
        height = image.shape[0]
        # return one point for each image corner which forces create_descriptors to randomly sample points around the center
        return [cv.KeyPoint(0, 0, 5, _class_id=0), cv.KeyPoint(width, 0, 5, _class_id=0), cv.KeyPoint(0, height, 5, _class_id=0), cv.KeyPoint(width, height, 5, _class_id=0)]

    def __str__(self):
        return "RANDOM"




