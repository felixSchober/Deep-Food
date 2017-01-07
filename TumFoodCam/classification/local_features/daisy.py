from classification.local_features.local_features import LocalFeatures
from classification.bag_of_words import BagOfWords
from data_io.testdata import TestData
import data_io.settings as Settings
from skimage.feature import daisy
from misc import utils
import cv2 as cv


class DaisyClassifier(LocalFeatures):
    """Deprecated DAISY classifier implementation [Tola2010]."""
    
    def __init__(self, testDataObj, svmType=None, name="", description=""):              
        super(DaisyClassifier, self).__init__(Settings.SI_TESTDATA_SEGMENTS, svmType, Settings.SI_SVM_PARAMS, testDataObj, name, True, description)            
        self.detector = None
        self.bow = BagOfWords(Settings.SI_BOW_DIMENSION, "SIFT")  


    def compute_descriptor(self, image):
        """
        This method overrides the compute_descriptor method from EdgeClassifier since skimage handles descriptor computation a little bit different.

        """
        descriptors = daisy(image, step=180, radius=58, rings=2, histograms=6,
                         orientations=8)
        #utils.display_images([image, img], None, ["Input", "DAISY descriptors"], 2, 1)        
        return (descriptors, descriptors) 

    def __str__(self):
        return "DAISY"


