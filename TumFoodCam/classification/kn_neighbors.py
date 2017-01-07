import os
import cv2 as cv
import numpy as np
from classification.model import FeatureModel
from classification.model import ModelType
import data_io.settings as Settings
from misc import utils
import data_io.settings as Settings
from math import exp
import logging

class KNearestNeighbours(FeatureModel):
    """ K Nearest Neighbors algorithm from OpenCV. Is currently used like svms in a one-vs-rest way. """
    
    
    def __init__(self, className):
        """
        Note: Set Settings.K_NUMBER_OF_NEIGHBORS_K for the variable k before calling this constructor.
        """
        super(KNearestNeighbours, self).__init__(ModelType.KNearest, True)      
        self.className = className
        self.__model = cv.KNearest()
        self.k = Settings.K_NUMBER_OF_NEIGHBORS_K

    def train(self, samples, labels):
        if samples[0].shape[0] == 1L:
            npsamples = np.concatenate(samples)
        else:
            npsamples = np.array(samples)

        # convert label list to numpy array
        nplabels = np.float32(labels)
        
        try:
            self.__model.train(npsamples, nplabels)
        except Exception, e:
            logging.exception("Could not train {0} samples.".format(len(labels)))            
            logging.info("\n\ndtype: {0}".format(npsamples.dtype))
            logging.info(npsamples)

    def predict(self, samples):
        # add dimension to the sample vectors because model.find_nearest expects a matrix in the form of (1L, nL) where n is the length of the previously trained feautre vector
        # without this the dimension would be (nL, )
        if any(s.shape[0] > 1 for s in samples):
            samples = [np.expand_dims(s, axis=0) for s in samples if s.shape[0] > 1]        

        results = [self.__model.find_nearest(s, k=self.k) for s in samples]
        retval, result, neighbors , dist = results[0]

        score = 0
        for i in xrange(self.k):
            # dist can be zero. To prevent zero divisions, make dist[i] small
            if dist[0][i] == 0.0:
                dist[0][i] = 0.1
            score += neighbors[0][i] * (1 / dist[0][i])

        #print "Retval: {0}, Results: {1}, Neigbors: {2}, DescFuncs: {3}, score: {4}".format(results[0][0], results[0][1], results[0][2], results[0][3], score)
        return [score]

    def save(self, path=utils.get_data_path(), fileName=None):
        if fileName is None:
            fileName = "knn_" + self.className
        path += fileName + ".xml"       
        #self.__model.save(path)

    def load(self, path=utils.get_data_path(), fileName=None):
        if fileName is None:
            fileName = "knn_" + self.className
        path += fileName + ".xml"       
        self.__model.load(path)

    def __getstate__(self):
        result = self.__dict__.copy()
        # delete the SVM because it can't be pickeled
        del result['_KNearestNeighbours__model']
        return result 

    def __setstate__(self, dict):
        self.__dict__ = dict
        # the trained SVM has to be restored with load manually 
        self.__model = cv.KNearest()


         


