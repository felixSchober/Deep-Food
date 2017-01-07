import time
import cv2 as cv
import numpy as np
from math import sqrt
import operator
import logging
from classification.feature_classifier import FeatureClassifier
from classification.bag_of_words import BagOfWords
from misc.model_tester import ModelTester, get_mean_accuracy
import data_io.settings as Settings
from misc import utils



class LocalFeatures(FeatureClassifier):
    """A super class for all OpenCV edge classifier like SIFT, SURF, ORB or BRISK."""

    def __init__(self, testDataSegments, svmType, svmParams, testData, name = '', grayscale = True, description=""):
        self.detector = None
        self.bow = None
        
        super(LocalFeatures, self).__init__(svmType, svmParams, testData, name, grayscale, description, imageSize=Settings.E_IMAGE_SIZE)
        
        #Segment the testData in three parts (2 train, 1 test)
        self.testData.segment_test_data(testDataSegments)
        self.transform = None
        self.numberOfExtractedKeypoints = []

    def restore_bow(self, path):
        """ Restores a specific bag of words by restoring its vocabulary. """

        if not utils.check_if_file_exists(path):
            print "Could not find bow. Path: {0}".format(path)
            return False
        self.bow = self.bow.load(path)
        self.bowTrained = True
        print "BOW successfully restored."
        return True

    def train_and_evaluate(self, save=True):
        """
        Trains and evaluates a feature detector/descriptor & BOW & SVM/KNN approach.
        """
        
        # load bow if path is provided
        if self.bowTrained:
            self.testData.new_segmentation()
            self.create_SVMs()
            self.trained = True
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
                
            testAccuracy = results["test"][0]
            testLoss = results["test"][1]
            iterationResults.append(results)
                
                
            row = [self.testData.crossValidationIteration, testLoss, testAccuracy, trainSVMLoss, trainSVMAccuracy, computeTime]
            accuracies.append(row)
            if Settings.G_EVALUATION_DETAIL_HIGH:
                confMatrices.append(self.tester.compute_confusion_matrix())
            print "{0}/{1} Cross Validation Iteration: Accuracy: {2}".format(self.testData.crossValidationIteration, self.testData.crossValidationLevel, testAccuracy)




        else:
            # create new from scratch + Cross validation

            # create classifier tester to evaluate the results from the cross validation runs.
            self.tester = ModelTester(self, transformation=self.transform)

            accuracies = []
            confMatrices = []
            iterationResults = []

            while self.testData.new_segmentation():  
                start = time.clock()          
                imageDescriptors = self.create_descriptors()
                self.bow.create_BOW(imageDescriptors)

                # save bow so that we can load it later.
                # this is redundant because we save the bow a second time, when we
                # save the whole model but this allows us to use the bow with a different SVM.
                if save:
                    self.bow.save(self.modelSaver)
                self.bowTrained = True
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
                
                testAccuracy = results["test"][0]
                testLoss = results["test"][1]
                iterationResults.append(results)
                
                
                row = [self.testData.crossValidationIteration, testLoss, testAccuracy, trainSVMLoss, trainSVMAccuracy, computeTime]
                accuracies.append(row)
                if Settings.G_EVALUATION_DETAIL_HIGH:
                    confMatrices.append(self.tester.compute_confusion_matrix())
                print "{0}/{1} Cross Validation Iteration: Accuracy: {2}".format(self.testData.crossValidationIteration, self.testData.crossValidationLevel, testAccuracy)

            header = ["Iteration", "test error", "test accuracy", "train SVM error", "train SVM accuracy", "compute time"]
            
        self.show_evaluation_results(accuracies, confMatrices, header)
        if save:  
            self.export_evaluation_results(iterationResults, confMatrices)
            self.save()
        # mean number of keypoints
        
        print "\nTraining of {0} done."
        keypointStats = np.array(self.numberOfExtractedKeypoints)
        print utils.get_table(["# m. filtered kp", "# total unfiltered kp", "m. unfiltered kp", "std. unfiltered kp"], 2, [Settings.E_NUMBER_OF_KEYPOINTS, keypointStats.sum(), keypointStats.mean(), keypointStats.std()])
        raw_input("Press any key to continue")
        return get_mean_accuracy(accuracies)
        
    def create_descriptors(self):
        """
        Creates initial descriptors needed for the BoW by loading images in the trainSVM segment. 
        Returns: list of descriptors
        """
        # For testing purposes only load 15 classes and 20 images per class.
        descriptors = []
        totalNumberOfDescriptors = float(self.testData.numberOfClasses * self.testData.segmentSizeMean["trainSVM"])

        if Settings.G_DETAILED_CONSOLE_OUTPUT:
            print "\nCreating {0} {1} descriptors.".format(totalNumberOfDescriptors, str(self))
        for img, _ in self.testData.load_data("trainSVM", outputActions=False, resolutionSize=self.imageSize, transformation=self.transform):
            
            _, descriptor = self.compute_descriptor(img)
            # Add the descriptor to the other descriptors we have.
            if descriptor is None or len(descriptor) == 0:
                print "\n** Could not find descriptors and/or keypoints for Image. **\n"

                # save for failure analysis
                vPath = self.modelSaver.get_save_path_for_visualizations() + "/noKeyPoints/"
                utils.create_dir_if_necessary(vPath)
                fileName = utils.get_uuid() + ".jpg"
                cv.imwrite(vPath + fileName, img)
                continue
  
            descriptors.append(descriptor)   
            # clear to prevent messing up RAM.
            img = []

            # Print progress
            utils.show_progress(Settings.G_DETAILED_CONSOLE_OUTPUT, len(descriptors), totalNumberOfDescriptors, "Descriptor creation Progress:")

        return descriptors

    def detect_keypoints(self, image):
        """ 
        Default method to get the keypoints by calling the detect() method of the detector.
        Overwrite if API works differently.
        """
        return self.detector.detect(image, None)

    def descibe_keypoints(self, image, keypoints):
        """ 
        Default method to get the keypoint descriptors by calling the compute() method of the detector.
        Overwrite if API works differently.
        """
        return self.detector.compute(image, keypoints)  

    def compute_descriptor(self, image, E_NUMBER_OF_KEYPOINTS=Settings.E_NUMBER_OF_KEYPOINTS, E_RANDOM_SAMPLE=Settings.E_RANDOM_SAMPLE, E_SCORE_WEIGHTING=Settings.E_SCORE_WEIGHTING):
        """
        Method to compute descriptors from an image by sampling keypoints from the image using self.detect_keypoints().

        Keyword arguments:
        E_NUMBER_OF_KEYPOINTS -- Maximum number of keypoints to sample
        E_RANDOM_SAMPLE -- Augment keypoints if sampledKeypoints < E_NUMBER_OF_KEYPOINTS
        E_SCORE_WEIGHTING -- Weight keypoints by response and distance to center.
        """
        
        keypoints = self.detect_keypoints(image)
        self.numberOfExtractedKeypoints.append(len(keypoints))

        # if no keypoints were detected return an empty descriptor
        if len(keypoints) == 0:
            return ([], [])

        if E_RANDOM_SAMPLE:
            kpPositions = np.array([k.pt for k in keypoints])

        # weight keypoints with distance score and filter weak keypoints
        if E_SCORE_WEIGHTING:
            center = (image.shape[1]/2, image.shape[0]/2)
            kpResponses = np.array([kp.response for kp in keypoints])
            kpDistances = np.array([sqrt((kp.pt[0]-center[0])**2 + (kp.pt[1]-center[1])**2) for kp in keypoints])

            # scale responses and distances -> [0, 1]
            kpResponses = (kpResponses - np.min(kpResponses)) / (np.max(kpResponses) - np.min(kpResponses))
            kpDistances = (kpDistances - np.min(kpDistances)) / (np.max(kpDistances) - np.min(kpDistances))
            
            # filter keypoints
            kpScores = []

            for i in xrange(len(keypoints)):
                score = 1 - (kpDistances[i] * (1-kpResponses[i])) # high score -> good
                kpScores.append((score, keypoints[i]))
                #if score > E_KEYPOINT_THRESHOLD:
                #    filteredKeypoints.append(keypoints[i])

            # sort scores so that better keypoints (higher scores) are top
            kpScores = sorted(kpScores, key=operator.itemgetter(0), reverse=True)

            # select keypoints
            selectedPoints = []
            for score, kp in kpScores:
               
                selectedPoints.append(kp)
                
                # break if desired number of keypoints reached
                if len(selectedPoints) >= E_NUMBER_OF_KEYPOINTS:
                    break
            keypoints = selectedPoints 
            
        if len(keypoints) < E_NUMBER_OF_KEYPOINTS and E_RANDOM_SAMPLE:
            #Goal not reached. Sample around mean
            try:
                kpMean = kpPositions.mean(axis=0)
                kpStd = (kpPositions.std(axis=0)/2) # half std

                # make sure the std is not 0
                kpStd[0] = max(0.00001, kpStd[0])
                kpStd[1] = max(0.00001, kpStd[1])

                toSample = E_NUMBER_OF_KEYPOINTS - len(keypoints)
                xDist = np.random.normal(kpMean[0], kpStd[0], toSample)
                yDist = np.random.normal(kpMean[1], kpStd[1], toSample)
                newKeyPoints = zip(xDist, yDist)

                for nKp in newKeyPoints:                    
                    keypoints.append(cv.KeyPoint(nKp[0], nKp[1], 5, _class_id=0))
            except:
                logging.exception("Could not sample new keypoints")
            
       
        # no score weighting -> restrict keypoints only by E_NUMBER_OF_KEYPOINTS 
        else:
            if len(keypoints) > E_NUMBER_OF_KEYPOINTS:
                keypoints = keypoints[:E_NUMBER_OF_KEYPOINTS]

        return self.descibe_keypoints(image, keypoints)

    def create_feature_vector(self, image):
        """Computes a bag of words feature vector for the image."""

        keypoints, _ = self.compute_descriptor(image)
        return self.bow.compute_feature_vector(image, keypoints)     