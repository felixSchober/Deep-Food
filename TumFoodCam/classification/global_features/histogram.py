import time
import cv2 as cv
from misc import utils
from classification.feature_classifier import FeatureClassifier
from misc.color_space import ColorSpace
from misc.model_tester import ModelTester, get_mean_accuracy
from data_io.testdata import TestData
import numpy as np
from  math import sqrt
import data_io.settings as Settings
import logging


class HistogramClassifier(FeatureClassifier):
    """Histogram classifier class that uses color information for classification. """

    def __init__(self, testDataObj, svmType=None, name="", description=""):       
         
        super(HistogramClassifier, self).__init__(svmType, Settings.H_SVM_PARAMS, testDataObj, name, grayscale=False, description=description, imageSize=Settings.H_IMAGE_SIZE)                 
        self.svmType = svmType
        self.__svms = {}

        #Segment the testData in tow parts (1 train, 1 test)
        self.testData.segment_test_data(Settings.H_TESTDATA_SEGMENTS)

        # 2 is a special case. In this case part the image in two pieces vertically
        if Settings.H_IMAGE_SEGMENTS == 2:
            self.imageCols = 2
            self.imageRows = 1
        else:
            self.imageCols = self.imageRows = sqrt(Settings.H_IMAGE_SEGMENTS)


    def train_and_evaluate(self, save=True):
        # create classifier tester to evaluate the results from the cross validation runs.
        self.tester = ModelTester(self)
        iterationResults = []
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
                results = self.tester.test_classifier(["test"])
                trainSVMAccuracy = "?"
                trainSVMLoss = "?"
            iterationResults.append(results)
                
            testAccuracy = results["test"][0]
            testLoss = results["test"][1]
                
                
            row = [self.testData.crossValidationIteration, testLoss, testAccuracy, trainSVMLoss, trainSVMAccuracy, computeTime]
            accuracies.append(row)
            if Settings.G_EVALUATION_DETAIL_HIGH:
                confMatrices.append(self.tester.compute_confusion_matrix(export=False))
            print "{0}/{1} Cross Validation Iteration: Accuracy: {2}".format(self.testData.crossValidationIteration, self.testData.crossValidationLevel, testAccuracy)

        header = ["Iteration", "test error", "test accuracy", "train error", "train accuracy", "compute time"]
        self.show_evaluation_results(accuracies, confMatrices, header)
        if save:
            self.export_evaluation_results(iterationResults, confMatrices)
            self.save()
        print "\nTraining of {0} done.".format(self.name) 
        return get_mean_accuracy(accuracies)

    def create_feature_vector(self, image):
        """ Creates the feature vector out of histograms."""

        # convert image depending on desired color space for the histogram
        temp = []
        colors = ("b", "g", "r")
        try:
            if Settings.H_COLOR_SPACE == ColorSpace.HSV:            
                temp = cv.cvtColor(image, cv.COLOR_BGR2HSV)            
                colors = ("h", "s", "v")
            elif Settings.H_COLOR_SPACE == ColorSpace.RGB:
                temp = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                colors = ("r", "g", "b")
            else:
                temp = image
        except:
            logging.exception("Could not convert image to {0}.".format(Settings.H_COLOR_SPACE))
            cv.imshow("Error CV", image)
            utils.display_images([image], None, ["Error"], 1, 1)

        # split image into x different parts.
        imageParts = []
        height, width = temp.shape[:2]
        partHeight = int(height / self.imageRows)
        partWidth = int(width / self.imageCols)
        scaleMaxPossibleValue = partWidth * partHeight

        # modify height and width in case partHeight and partWidth don't add up to the real width and height.
        # in this case the image would be cut into more than x parts because some leftover pixels would be included.
        height = int(partHeight * self.imageRows)
        width = int(partWidth * self.imageCols)

        for y in xrange(0, height, partHeight):
            for x in xrange(0, width, partWidth):
                imageParts.append(utils.crop_image(temp, x, y, partWidth, partHeight))
                
        histogram = []
        for img in imageParts:
            for i, color in enumerate(colors):
                hist = cv.calcHist([img], [i], None, [Settings.H_BINS], Settings.H_COLOR_RANGE) 
                
                if Settings.H_SCALE_HIST:
                    # max possible value is w * h of imagePart
                    hist /= scaleMaxPossibleValue
                                   
                histogram.extend(hist)

        return np.array(np.concatenate(histogram))

    def __str__(self):
        return "HIST"