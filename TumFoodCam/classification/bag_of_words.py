import os
import pickle
import cv2 as cv
import numpy as np
from misc import utils
import data_io.settings as Settings
import logging

class BagOfWords(object):
    """Wrapper for openCV Bag of words logic."""

    PICKLE_CLASS_NAME = "BOW"
    
    def __init__(self, size, dextractor="SIFT", dmatcher="FlannBased", vocabularyType=np.float32):
        self.size = size
        self.dextractor = dextractor
        self.dmatcher = dmatcher
        self.vocabularyType = vocabularyType

    def create_BOW(self, descriptors):
        """Computes a Bag of Words with a set of descriptors."""

        print "\n\nCreating BOW with size {0} with {1} descriptors.".format(self.size, len(descriptors))
        bowTrainer = cv.BOWKMeansTrainer(self.size)

        # Convert the list of numpy arrays to a single numpy array
        npdescriptors = np.concatenate(descriptors)

        # an OpenCV BoW only takes floats as descriptors. convert if necessary
        if not npdescriptors.dtype == np.float32:
            npdescriptors = np.float32(npdescriptors)

        if Settings.G_DETAILED_CONSOLE_OUTPUT:
            print "Clustering BOW with Extractor {0} and Matcher {1}".format(self.dextractor, self.dmatcher)
        self.__BOWVocabulary = bowTrainer.cluster(npdescriptors)

        # need to convert vocabulary?
        if self.__BOWVocabulary.dtype != self.vocabularyType:
            self.__BOWVocabulary = self.__BOWVocabulary.astype(self.vocabularyType)

        print "BOW vocabulary creation finished."

        # Create the BoW descriptor
        self.__BOWDescriptor = cv.BOWImgDescriptorExtractor(cv.DescriptorExtractor_create(self.dextractor), cv.DescriptorMatcher_create(self.dmatcher))
        self.__BOWDescriptor.setVocabulary(self.__BOWVocabulary)

    def compute_feature_vector(self, image, keypoints):
        return self.__BOWDescriptor.compute(image, keypoints)

    def save(self, modelSaver):
        path = modelSaver.get_save_path()
        with open(path + "BOW", "wb") as f:
            pickle.dump(self,f)

    def load(self, path):        
        with open(path, "r") as f:
            self = pickle.load(f)
                    
        return self

    def __getstate__(self):
        result = self.__dict__.copy()
        del result['_BagOfWords__BOWDescriptor']
        return result 

    def __setstate__(self, dict):
        self.__dict__ = dict
        # restore __BOWDescriptor we had to remove
        self.__BOWDescriptor = cv.BOWImgDescriptorExtractor(cv.DescriptorExtractor_create(self.dextractor), cv.DescriptorMatcher_create(self.dmatcher))
        self.__BOWDescriptor.setVocabulary(self.__BOWVocabulary)

        


