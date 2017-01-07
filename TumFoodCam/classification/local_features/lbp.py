import time
from misc import utils
from skimage import feature
from classification.feature_classifier import FeatureClassifier
from misc.model_tester import ModelTester, get_mean_accuracy
from data_io.testdata import TestData
import numpy as np
import cv2 as cv
import data_io.settings as Settings
import time


class LBPClassifier(FeatureClassifier):
    """A basic local binary pattern classifier [65]."""
    

    def __init__(self, testDataObj, svmType=None, name="", description=""):
        
        super(LBPClassifier, self).__init__(svmType, Settings.L_SVM_PARAMS, testDataObj, name, grayscale=True, description=description)                 
        self.svmType = svmType
        self.__svms = {}

        #Segment the testData in tow parts (1 train, 1 test)
        self.testData.segment_test_data(Settings.L_TESTDATA_SEGMENTS)

    def train_and_evaluate(self, save=True):
        # create classifier tester to evaluate the results from the cross validation runs.
        self.tester = ModelTester(self)
        accuracies = []
        confMatrices = []
        while self.testData.new_segmentation():
            start = time.clock()          
            self.create_SVMs()
            self.trained = True
            computeTime = time.clock() - start

            # evaluate
            if Settings.G_EVALUATION_DETAIL_HIGH:
                results = self.tester.test_classifier(["trainSVM", "test"])
                #extract interesting results
                trainSVMAccuracy = results["trainSVM"][0]
                trainSVMLoss = results["trainSVM"][1]
                    
            else:
                trainSVMAccuracy = "?"
                trainSVMLoss = "?"
                
            testAccuracy = results["test"][0]
            testLoss = results["test"][1]
                
                
            row = [self.testData.crossValidationIteration, testLoss, testAccuracy, trainSVMLoss, trainSVMAccuracy, computeTime]
            accuracies.append(row)
            if Settings.G_EVALUATION_DETAIL_HIGH:
                confMatrices.append(self.tester.compute_confusion_matrix())
            print "{0}/{1} Cross Validation Iteration: Accuracy: {2}".format(self.testData.crossValidationIteration, self.testData.crossValidationLevel, testAccuracy)

        header = ["Iteration", "test error", "test accuracy", "train error", "train accuracy", "compute time"]

        self.show_evaluation_results(accuracies, confMatrices, header)
        if save:
            self.save()
        print "\nTraining of {0} done.".format(self.name) 
        return get_mean_accuracy(accuracies)

    def create_feature_vector(self, image):
        pattern = feature.local_binary_pattern(image, Settings.L_NUMBER_OF_POINTS, Settings.L_RADIUS, Settings.L_METHOD)

        # create a histogram of the pattern because pattern is a 2D array with image.shape dimension.
        # values inside pattern range from [0, L_NUMBER_OF_POINTS + 2]
        # there is a value for each L_NUMBER_OF_POINTS + 1 "possible rotation invariant prototypes" (since we use method = uniform)
        # + extra dim for all possible not uniform values
        
        #convert to 32 float
        pattern = np.float32(pattern)
        hist = cv.calcHist([pattern.ravel()], [0], None, [Settings.L_NUMBER_OF_POINTS + 2], [0, Settings.L_NUMBER_OF_POINTS + 1])
          
        # normalize histogram
        hist /= (hist.sum() + Settings.L_EPS)
        return np.array(hist, dtype=np.float32)      

    def __str__(self):
        return "LBP"

            


