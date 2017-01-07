import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import cv2 as cv
import numpy as np
import logging
from enum import Enum
import random
import time
from classification.model import ModelType
from classification.local_features.sift import SIFTClassifier
from classification.local_features.surf import SURFClassifier
from classification.local_features.orb import ORBClassifier
from classification.late_fusion import MultiStepClassifier, MajorityClassifier
from classification.global_features.histogram import HistogramClassifier
from classification.local_features.lbp import LBPClassifier
from classification.local_features.random_keypoint_sampler import RandomKeypointSampler
#from classification.local_features.daisy import DaisyClassifier
#from classification.deep.neural_net import NeuralNetClassifier
import data_io.settings as Settings
from data_io.settings_IO_service import SettingsService
from data_io.testdata import TestData
from matplotlib import pyplot as plt
from misc import utils
from data_io.model_IO import ModelSaver, CsvExporter, ModelLoader

class ClassifierType(Enum):
    SIFT = 1
    HIST = 2
    SURF = 3
    ORB = 4
    DAISY = 5
    CENSURE = 6
    LF = 7
    LBP = 8
    NN = 9
    VOTE = 10
    RANDOM = 11



class ModelController(object):
    """
    Class to control (test, train, predict) a model.
    DO NOT instantiate this class but use MODEL_CONTROLLER from this model.
    """
    
    def __init__(self):
        self.modelLoader = ModelLoader()

    def __instantiate_model_by_id(self, name=None, description=None):
        """ 
        Instantiates a model using self.__modelId of the type ClassifierType
        
        Keyword arguments:
        name -- Name of the model. Default: Ask
        description -- Description of the model. Default: Ask 
        """
        if name is None:
            name = utils.value_question("[?]", "Provide a name for the model", "s")
        if description is None:
            description = utils.value_question("[?]", "Provide a description for the model", "s", "(optional)", True)
            
        # Ini testdata
        if self.__modelId != ClassifierType.VOTE.value:
            testdata = TestData(Settings.G_TEST_DATA_PATH, Settings.H_CROSS_VALIDATION_K, True)

        # Ini potential bow
        
            
        if self.__modelId == ClassifierType.SIFT.value:                  
            m = SIFTClassifier(testdata, Settings.E_MODEL_TYPE, name, description)
            bow = utils.value_question("[?]", "Provide a precomputed bow", "s", "(optional)", True)
            if bow != "":
                m.restore_bow(bow)
            return m

        elif self.__modelId == ClassifierType.HIST.value:
            return HistogramClassifier(testdata, Settings.H_MODEL_TYPE, name, description)

        elif self.__modelId == ClassifierType.SURF.value:
            m = SURFClassifier(testdata, Settings.E_MODEL_TYPE, name, description)
            bow = utils.value_question("[?]", "Provide a precomputed bow", "s", "(optional)", True)
            if bow != "":
                m.restore_bow(bow)
            return m

        elif self.__modelId == ClassifierType.ORB.value:
            return ORBClassifier(testdata, Settings.E_MODEL_TYPE, name, description)

        elif self.__modelId == ClassifierType.DAISY.value:
            return DaisyClassifier(testdata, Settings.E_MODEL_TYPE, name, description)

        elif self.__modelId == ClassifierType.CENSURE.value:
            from classification.local_features.censure import CENSUREClassifier
            return CENSUREClassifier(testdata, Settings.E_MODEL_TYPE, name, description)

        elif self.__modelId == ClassifierType.LF.value:
            return MultiStepClassifier(testdata, ModelType.OpenCV, name, description)

        elif self.__modelId == ClassifierType.LBP.value:
            return LBPClassifier(testdata, Settings.L_MODEL_TYPE, name, description)

        elif self.__modelId == ClassifierType.NN.value:
            return NeuralNetClassifier(testdata, name, description=description)

        elif self.__modelId == ClassifierType.VOTE.value:
            return MajorityClassifier(name, description)
        elif self.__modelId == ClassifierType.RANDOM.value:
            return RandomKeypointSampler(testdata, Settings.E_MODEL_TYPE, name, description)

    def show_model_screen(self):
        """
        Prints a model selection screen to the console window.

        Returns:
        ModelId
        """

        utils.clear_screen()
        options = ["SIFT kp/dp + BoW + SVM", "Histogram & SVM", "SURF kp/dp + BoW + SVM", "ORB kp/dp + BoW + SVM", "DAISY [deprecated]", "CenSurE kp, SIFT dp + BoW + SVM", "Late Fusion - SVM-Vectors", "LBP + SVM", "Neural Nets", "Late Fusion Majority Vote", "Random Keypoint Sampling"]
        modelId = utils.menue("Choose model:", options, False)
        if modelId == len(options):
            return -1
        return modelId

    def set_model(self, modelId, name=None, description=None):
        """ 
        Sets a model.
        """

        self.__modelId = modelId        
        self.model = self.__instantiate_model_by_id(name, description)
    
    def reset_model(self):
        """ Resets a model by instantiating it again."""
        name = "?"
        description = ""
        if self.has_model:
            name = self.model.name
            description = self.model.description
        self.model = self.__instantiate_model_by_id(name, description)

    def load_model(self):
        """ Loads a model using the loading screen."""
        model = self.modelLoader.show_loading_screen()
        if model is None:
            return False
        self.model = model
        return True

    def run_and_test(self, testMode):
        """
        Train and test the model.
        
        Keyword arguments:
        testMode -- enables test mode for model training.
        
        Returns: return value of train_and_evaluate (usually accuracy)
        """
        return self.model.train_and_evaluate((not testMode))

    @property
    def has_model(self):
        """ Boolean if ModelController has a model."""
        return not self.model is None 

    @property
    def has_tester(self):
        """ Boolean if model has a ModelTester instance."""
        return self.has_model and not self.model.tester is None

    @property
    def tester(self):
        """ ModelTester instance. Returns None if has_tester is False."""
        if self.has_tester:
            return self.model.tester
        return None

MODEL_CONTROLLER = ModelController()

class ParameterTester(object):

    def __init__(self, testRunName, testRunDescription, modelId, testSuite):
        MODEL_CONTROLLER.set_model(modelId, testRunName, testRunDescription)
        self.testSuite = testSuite
        self.modelSaver = ModelSaver(MODEL_CONTROLLER.model.testData.get_root_path(), testRunName, testRunDescription)

    def start_testing(self):
        Settings.G_DETAILED_CONSOLE_OUTPUT = False
        parameterList = self.testSuite.parameter_list
        title = ["iteration"]
        title.extend(parameterList)
        title.extend(["accuracy", "time"])
        data = [title]
        accuracies = []
        for iteration in xrange(self.testSuite.iterations):

            values = self.testSuite.set_values(iteration)

            print "\n\n** Iteration {0} - Value: {1} **\n\n".format(iteration, values)
            # reset with new value
            MODEL_CONTROLLER.reset_model()
            start = time.clock()     
            accuracy = MODEL_CONTROLLER.run_and_test(True)
            computeTime = time.clock() - start
            accuracies.append(accuracy)
            # export
            iterationResults = [iteration]
            for param in parameterList:
                iterationResults.append(values[param])
            iterationResults.append(accuracy)
            iterationResults.append(computeTime)
            data.append(iterationResults)
            self.modelSaver.get_csv_exporter().export(data, "report")

            # show
            print utils.get_table(title, 4, iterationResults)

            

        # plot 
        try:
            accuracies = np.array(accuracies)
            acMean = accuracies.mean()
            acStd = accuracies.std()
            acLower = acMean-2*acStd
            acUpper = acMean+2*acStd
            plt.clf()
            plt.figure(figsize=(12, 9))
            ax = plt.subplot(111)  
            ax.spines["top"].set_visible(False)  
            ax.spines["right"].set_visible(False) 
            ax.get_xaxis().tick_bottom()  
            ax.get_yaxis().tick_left()   
            #plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14) 

            plt.ylabel("Accuracy", fontsize=16)
            plt.title("Accuracy results of {0}".format(MODEL_CONTROLLER.model.name), fontsize=22)  

            # if only one parameter -> plot line
            if len(parameterList) == 1:
                plt.xlabel("Value of {0}".format(parameterList), fontsize=16)
                plt.axis([self.paramValues[0], self.paramValues[-1], acLower, acUpper])                
                x = self.paramValues
            else:
                plt.xlabel("Iteration", fontsize=16)
                plt.axis([0, self.testSuite.iterations, acLower, acUpper])
                x = range(self.testSuite.iterations)
            plt.plot(x, accuracies, color=Settings.G_COLOR_PALETTE[0])
            if len(accuracies) <= 8:
                utils.annotate_points(ax, x, accuracies)   
            plt.show()
            utils.save_plt_figure(plt, "testingResults", self.modelSaver.get_save_path_for_visualizations())
        except Exception, e:
            logging.exception("Could not plot/show results. Trying to save instead.")                
            utils.save_plt_figure(plt, "testingResults", self.modelSaver.get_save_path_for_visualizations())
                    
        # restore settings
        settingsService = SettingsService()
        settingsService.load()


class Testsuite(object):

    def __init__(self, iterations=-1, random=False, seed=42):
        """ Iterations = -1 means that the number of iterations should be determined by the number of parameterValues
        """
        self.parameters = {}
        self.iterations = iterations
        self.random = random
        self.seed = seed


    def add_parameter(self, paramId, paramName, paramValues, paramInterval=None, discreteValues=True):
        if not self.random and self.iterations > len(paramValues):
            raise AttributeError("Testsuite is not random and the number of iterations is bigger than the avaiable parameters for {0}.".format(paramName))
        values = []
        if self.random:
            #random.seed(self.seed)

            
            if paramInterval is None:
                # allow repeating
                values = [paramValues[random.randint(0, len(paramValues) - 1)] for _ in xrange(self.iterations)]

            # values from an interaval
            else:
                if discreteValues:
                    values = random.sample(xrange(paramInterval[0], paramInterval[1]), self.iterations)
                else:
                    values = [random.uniform(paramInterval[0], paramInterval[1]) for _ in xrange(self.iterations)]
        else:
            values = paramValues
            self.iterations = len(values)

        self.parameters[paramId] = (paramName, values)

    @property
    def parameter_list(self):
        return [param[0] for param in self.parameters.values()]

    def set_values(self, iteration):
        iterationSetting = {} # for visualization
        for paramId in self.parameters:
            iterationSetting[self.parameters[paramId][0]] = self.parameters[paramId][1][iteration]
            self.__set_parameter(paramId, self.parameters[paramId][1][iteration])
        return iterationSetting

    def __set_parameter(self, paramId, value):
        if paramId == 0:
            Settings.H_COLOR_SPACE = value
        elif paramId == 1:
            Settings.H_IMAGE_SEGMENTS = value
        elif paramId == 2:
            Settings.H_BINS = value
        elif paramId == 3:
            Settings.H_SVM_PARAMS["degree"] = value
        elif paramId == 4:
            Settings.E_NUMBER_OF_KEYPOINTS = value
        elif paramId == 5:
            Settings.SI_BOW_DIMENSION = value
        elif paramId == 6:
            Settings.SU_BOW_DIMENSION = value
        elif paramId == 7:
            Settings.E_SCORE_WEIGHTING = value
        elif paramId == 8:
            Settings.F_APPLY_ZNORMALIZATION = False
            Settings.F_APPLY_ZCA_WHITENING = False
            Settings.F_APPLY_CLAHE = False
            Settings.F_APPLY_HISTOGRAM_EQ = False
            Settings.F_APPLY_NORMALIZATION = value
        elif paramId == 9:
            Settings.F_APPLY_ZNORMALIZATION = False
            Settings.F_APPLY_ZCA_WHITENING = False
            Settings.F_APPLY_CLAHE = value
            Settings.F_APPLY_HISTOGRAM_EQ = False
            Settings.F_APPLY_NORMALIZATION = False
        elif paramId == 10:
            Settings.E_IMAGE_SIZE = value
        elif paramId == 11:
            Settings.E_MODEL_TYPE = value
        elif paramId == 12:
            Settings.H_MODEL_TYPE = value
        elif paramId == 13:
            Settings.E_MODEL_TYPE = ModelType.OpenCV
            Settings.SU_SVM_PARAMS = value
        elif paramId == 14:
            Settings.L_SVM_PARAMS = value
        elif paramId == 15:
            Settings.C_SVM_PARAMS = value
        elif paramId == 16:
            Settings.H_MODEL_TYPE = ModelType.OpenCV
            Settings.H_SVM_PARAMS = value
        elif paramId == 17:
            Settings.E_MODEL_TYPE = ModelType.OpenCV
            Settings.SI_SVM_PARAMS = value
        elif paramId == 18:
            Settings.E_MODEL_TYPE = ModelType.OpenCV
            Settings.O_SVM_PARAMS = value
        elif paramId == 19:
            Settings.K_NUMBER_OF_NEIGHBORS_K = value
            Settings.H_MODEL_TYPE = ModelType.KNearest
            Settings.E_MODEL_TYPE = ModelType.KNearest
            Settings.L_MODEL_TYPE = ModelType.KNearest
        elif paramId == 20:
            Settings.L_SVM_PARAMS = ModelType.OpenCV
            Settings.L_SVM_PARAMS = value
        elif paramId == 21:
            Settings.H_IMAGE_SIZE = value






                

