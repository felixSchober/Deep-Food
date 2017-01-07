from misc import utils
from enum import Enum

class Model(object):
    """Abstract super class for different models implementations. (SVM, KNN, NN)"""
    def __init__(self):
        self.trained = False
        
        
    def train(self, samples, labels):
        """ 
        Method to train the model. Samples and labels need to have the same length.

        Keyword arguments:
        samples -- list or numpy array of samples.
        labels -- list of labels for samples.
        """

        raise NotImplementedError("abstract class")

    def predict(self, samples):
        """ Method to predict a list of samples."""
        raise NotImplementedError("abstract class")

    def save(self, path=utils.get_data_path(), fileName=None):
        """ Method to save a svm."""
        raise NotImplementedError("abstract class")

    def load(self, path=utils.get_data_path(), fileName=None):
        """ Method to load a svm."""
        raise NotImplementedError("abstract class")

class FeatureModel(Model):
    """Abstract superclass for SVM and KNN implementations."""

    def __init__(self, modelType, reverseSorting):
        super(FeatureModel, self).__init__()
        self.type = modelType
        self.trained = False
        self.reverseSorting = reverseSorting


class ModelType(Enum):
    OpenCV = 1
    Sklearn = 4
    KNearest = 5
    NN = 6





