import cv2 as cv
import numpy as np
from classification.feature_classifier import FeatureClassifier
from classification.local_features.sift import SIFTClassifier
from classification.global_features.histogram import HistogramClassifier
from classification.local_features.surf import SURFClassifier
#from classification.deep.neural_net import *
from data_io.testdata import TestData
from misc.model_tester import ModelTester
import data_io.settings as Settings
from data_io.model_IO import ModelSaver, ModelLoader
from misc import utils
import operator

class MultiStepClassifier(FeatureClassifier):
    """
    Deprecated: A 'bad' late fusion kind of implementation. DO NOT use!
    This class concatenates the outputs of multiple models together and predicts on these outputs.
    """


    def __init__(self, testDataObj, svmType=None, name="", description=""):
        self.name = name
        self.trained = False
        self.svms = {}
        self.svmParams = Settings.M_SVM_PARAMS

        self.__classifiers = [SIFTClassifier(svmType, "MS_SIFT_" + name), SURFClassifier(svmType, "MS_SURF_" + name), HistogramClassifier(svmType, "MS_HIST_" + name)]
        #self.__classifiers = [HistogramClassifier(svmType, "MS_HIST_" + name)]

        super(MultiStepClassifier, self).__init__(svmType, Settings.M_SVM_PARAMS, testDataObj, name, grayscale=False, description=description)                 
        #Segment the testData in three parts (2 train, 1 test)
        self.testData.segment_test_data({"trainSVM": 0.8, "test": 0.2})


    def restore_classifier(self):
        # does the directory exist?
        path = utils.get_data_path() + self.name + "/"
        if not utils.check_if_dir_exists(path):
            return False
        
        # iterate over __classifiers and try to restore them using the name
        for classifierIndex in xrange(len(self.__classifiers)):
            # check classifier folder
            classifier = self.__classifiers[classifierIndex]
            classifierPath = utils.get_data_path() + classifier.name + "/"
            if not utils.check_if_dir_exists(classifierPath):
                return False
            self.__classifiers[classifierIndex] = classifier.load(classifierPath)
            print "*  loaded {0} {1} including {2} SVMs.".format(str(classifier), classifier.name, len(self.__classifiers[classifierIndex].svms))
        print "** Successfully loaded all classifiers.\n"
        return True

    def train_and_evaluate(self, save=True):        
        # Step 1: train the classifiers independently
        print "Starting training of classifiers"
        for classifier in self.__classifiers:
            if classifier.trained and len(classifier.svms) == self.testData.numberOfClasses:
                print "{0} already trained - SKIP".format(str(classifier))
                continue
            classifier.train_and_evaluate(save)

        # Step 2: call create SVMs. This method calls create_feature_vector
        # create classifier tester to evaluate the results from the cross validation runs.
        self.tester = ModelTester(self)
        accuracies = []
        confMatrices = []
        while self.testData.new_segmentation():
            self.create_SVMs()
            self.trained = True

            accuracy, _ = self.tester.test_classifier()
            accuracies.append(accuracy)
            if Settings.G_EVALUATION_DETAIL_HIGH:
                confMatrices.append(self.tester.compute_confusion_matrix())
            print "{0}/{1} Cross Validation Iteration: Accuracy: {2}".format(self.testData.crossValidationIteration, self.testData.crossValidationLevel, accuracy)

        self.show_evaluation_results(accuracies, confMatrices)
        if save:
            self.save()
        print "\nTraining of {0} done.".format(self.name) 
        
    def create_feature_vector(self, image):
        predictionVector = []
        for classifier in self.__classifiers:
            prediction = classifier.predict([image], False)
            predictionVector.extend([prediction[k][0] for k in prediction])
        return np.array(predictionVector, dtype=np.float32)


class MajorityClassifier(FeatureClassifier):
    """
    Majority Vote classifier. 
    Loads multiple classifiers and predicts the class with the most votes without a learned weighting.
    """

    def __init__(self, name, description=""):
        self.classifiers = []
        self.modelSaver = ModelSaver(Settings.G_TEST_DATA_PATH, name, description)
        self.grayscale = False
        self.size = (-1, -1)
        self.name = name
        self.imageSize = 160000

    def initialize(self):
        """ Method to load the classifiers and test them."""

        loader = ModelLoader()

        input = None
        testDataPath = None
        while True:
            input = utils.menue("Majority Vote classifier", ["add classifier", "save", "test + finish", "finish"], False, True)
            if input == 2:
                self.modelSaver(self, -1)
                continue

            if input > 2:
                break

            # Display loading menu
            model = loader.show_loading_screen()
            if not model is None:
                if testDataPath is None:
                    testDataPath = model.testData.get_root_path()
                    self.modelSaver.datasetPath = testDataPath
                else:
                    # check if the test datasets are the same
                    if testDataPath != model.testData.get_root_path():
                        print "Could not load classifier {0} because the classifier was trained on different test data.".format(model.name)
                        continue
                self.classifiers.append((1, model)) # Tuple[0] = model weight (1 is default weight) | Tuple[1] = model

        # why did we leave the loop?
        if input == 5:
            print "Cancel"
            return False # cancel -> back to start
        else:
            # initialize test data for the majority classifier

            # check if test data path has changed
            if not utils.check_if_dir_exists(testDataPath):
                testDataPath = utils.value_question("[...]", "Root path to new Dataset Path", "s")
            self.testData = TestData(testDataPath, 1, False)
            self.testData.segment_test_data({"test": 1})
            self.testData.new_segmentation()

            self.tester = ModelTester(self)

            # test classifier if input == 3
            if input == 3:
                # Be careful: the results might not reflect the actual accuracy of the classifier.
                # if not changed the tester will test on the whole test data set. This might include images that the 
                # classifiers has been trained on. For a real accuracy test the images have to be separated manually.
                results = self.tester.test_classifier(["test"])
                self.tester.save_results(results, exportToCSV=False)
                print self.tester.format_results_string(results)
                testLoss = results["test"][1]
                save = utils.radio_question("[?]", "Save/Update classifier?", None, ["Yes", "No"], [True, False])
                if save:
                    self.modelSaver(self, testLoss)


            return self.classifiers != [] # finish. Return True if classifiers where loaded, False if not.

    def train_and_evaluate(self, save = True):
        self.initialize()

    def predict(self, images):
        for img in images:
            
            predictions = {class_: 0 for class_ in self.testData.classes}
            for clWeight, classifier in self.classifiers:
                # copy img in case we need to do some preprocessing
                clImg = np.copy(img)
                # grayscale or color
                if classifier.grayscale:                    
                    clImg = cv.cvtColor(clImg, cv.COLOR_BGR2GRAY)        

                # is there a size constraint for the classifier?
                size = classifier.tester.size
                if size != (-1, -1):
                    clImg = utils.crop_to_square(clImg)
                    desiredArea = size[0] * size[1]
                    clImg = utils.equalize_image_size(clImg, desiredArea)

                # do we need to transform the image for the classifier?
                transform = classifier.tester.transformation
                if not transform is None:
                    clImg = transform(clImg)

                clPrediction = classifier.predict([clImg])[:5] # get top-5 predictions
                vote = 5
                for class_, pValue in clPrediction:
                    predictions[class_] += vote * clWeight
                    vote -= 1
            # convert predictions dict back to list and sort it             
            pList = sorted([(class_, [votes]) for class_, votes in predictions.iteritems()], key=operator.itemgetter(1), reverse=True)
            return pList

    def save(self, path):
        # instead of the models, save the modelUuids
        self.classifierUuids = []
        for clWeight, cl in self.classifiers:
            self.classifierUuids.append((clWeight, cl.modelSaver.modelUuid))

    def __getstate__(self):
        result = self.__dict__.copy()

        # instead of the models, save the modelUuids
        classifierUuids = []
        for clWeight, cl in self.classifiers:
            classifierUuids.append((clWeight, cl.modelSaver.modelUuid))

        result["classifiers"] = classifierUuids
        return result 

    def __setstate__(self, dict):
        # try to restore all classifiers by loading them
        ml = ModelLoader()
        classifiers = []
        for clWeight, clUuid in dict["classifiers"]:
            try:    
                classifiers.append((clWeight, ml.load_model(clUuid)))
            except:
                raise Exception("Could not load classifier because at least one model ({0}) was not found.".format(clUuid))
        dict["classifiers"] = classifiers        
        self.__dict__ = dict

    def __str__(self):
        selfRepr = "mCl"
        for classifier, _ in self.classifiers:
            selfRepr += " " + str(classifier)
        return selfRepr
           



