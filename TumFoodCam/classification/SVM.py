import os
import cv2 as cv
import numpy as np
from misc import utils
from classification.model import FeatureModel, ModelType
#from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel, additive_chi2_kernel
import data_io.settings as Settings
from math import exp
import logging


class OpenCvSVM(FeatureModel):
    """
    OpenCV SVM with 3 possible Kernels:
        - Linear
        - Poly
        - RBF
    """

    def __init__(self, className, params):
        """ 
        Constructor for OpenCvSVM.
        
        Keyword arguments:
        className -- Name of the SVM image class.
        params -- Parameters for the OpenCV SVM. Parameters have to be a dictionary in accordance to http://docs.opencv.org/2.4/modules/ml/doc/support_vector_machines.html#cvsvmparams-cvsvmparams. A possible configuration is dict(kernel_type=cv.SVM_POLY, svm_type=cv.SVM_C_SVC, C=5, degree=2).
  
        """
        super(OpenCvSVM, self).__init__(ModelType.OpenCV, True)      
        self.className = className
        self.__model = cv.SVM()
        self.params = params

    def train(self, samples, labels):
        # reduce (if necessary) from 1L, X to X
        if samples[0].shape[0] == 1L:
            npsamples = np.concatenate(samples)
        else:
            npsamples = np.array(samples)

        # convert label list to numpy array
        nplabels = np.float32(labels)

        try:
            self.__model.train(npsamples, nplabels, params=self.params)
        except Exception, e:
            logging.exception("Could not train {0} samples.".format(len(labels)))            
            logging.info("\n\ndtype: {0}".format(npsamples.dtype))
            logging.info(npsamples)

    def predict(self, samples):
        """ 
        predict returns the distance to the decision function
        if p < 0 this means a positive output (the SVM predicts that the sample is positive)
        if p > 0 the sample is negative.
        """

        predictions = [self.__model.predict(s, True) for s in samples]
        
        # calculate confidences
        confidences = []
        
        for p in predictions:
            confidences.append((p * -1))

        return confidences

    def save(self, path=utils.get_data_path(), fileName=None):
        if fileName is None:
            fileName = "svm_" + self.className
        path += fileName + ".xml"       
        self.__model.save(path)

    def load(self, path=utils.get_data_path(), fileName=None):
        if fileName is None:
            fileName = "svm_" + self.className
        path += fileName + ".xml"       
        self.__model.load(path)

    def __getstate__(self):
        result = self.__dict__.copy()
        # delete the SVM because it can't be pickeled
        del result['_OpenCvSVM__model']
        return result 

    def __setstate__(self, dict):
        self.__dict__ = dict
        # the trained SVM has to be restored with load manually 
        self.__model = cv.SVM()


class SklearnSVM(FeatureModel):
    """
    Sklearn SVM with additive Chi squared kernel.
    """

    def __init__(self, className):
        super(SklearnSVM, self).__init__(ModelType.Sklearn, True)      
        self.className = className

        self.__model = None

    def train(self, samples, labels):
        if samples[0].shape[0] == 1L:
            npsamples = np.concatenate(samples)
        else:
            npsamples = np.array(samples)

        # make sure that npsamples have the correct shape
        npsamples = npsamples.reshape((-1, npsamples.shape[1]))

        # convert label list to numpy array
        nplabels = np.float32(labels)
        self.npsamples = npsamples        
        try:
            # map to Chi^2 kernel
            k = additive_chi2_kernel(npsamples)
            self.__model = SVC(kernel="precomputed").fit(k, labels)
        except:
            logging.exception("Could not train {0} samples.".format(len(labels)))
            print "\n\ndtype:",npsamples.dtype
            raw_input("Press any key to continue")
            print npsamples

    def predict(self, samples):
        # transform data
        samples = np.array(samples)
        samples = samples.reshape(1, -1)
        sample = additive_chi2_kernel(samples, self.npsamples)
        #samples = [chi2_kernel(s.reshape(1, -1)) for s in samples]

        predictions = self.__model.decision_function(sample)
        predictionValues = self.__model.predict(sample)
        # calculate confidences
        confidences = []
        
        for p in predictions:
            confidences.append((p * 1))

        return confidences

    def save(self, path = utils.get_data_path(), fileName = None):
        print "No pickle support. Use sklearn save method."




