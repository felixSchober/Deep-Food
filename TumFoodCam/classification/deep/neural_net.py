import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import cPickle as pickle
import sys
import time
import operator
import logging

from data_io.model_IO import NeuralNetSaver, CsvExporter
from data_io.testdata import TestData
from misc.model_tester import ModelTester
from misc import utils
import data_io.settings as Settings

from classification.classifier import Classifier
from classification.deep.training_history import plot_train_history, TrainingHistory
from classification.deep.nets.net_nouri3_cnn import net_nouri3_cnn
from classification.deep.nets.net_nouri4_cnn import net_nouri4_cnn
from classification.deep.nets.net_nouri5_cnn import net_nouri5_cnn
from classification.deep.nets.net_nouri6_cnn import net_nouri6_cnn, net_nouri6_cnn_do, net_nouri6_cnn_do_color, net_nouri6_cnn_do2_color
from classification.deep.nets.net_512 import net_512
#from classification.deep.nets.net_nouri2_cnn import net_nouri2_cnn
from classification.deep.nets.net_cifar_cnn import net_cifar_cnn, net_cifar_cnn_do
from classification.deep.nets.net_100_nnSimple import net_100_nnSimple
from classification.deep.nets.net_krizhevsky_cnn import net_krizhevsky_cnn


# for large networks pickle will crash because we reach the maximum recursion
sys.setrecursionlimit(10000)

currentNet = net_nouri6_cnn_do2_color

def normalize_image(image):
    """
    This contains the following operations
      1. Rescaling:               Images are scaled from [0, 255] to [0, 1]
    
    Deprecated:
    ( 2. Mean subtraction         The mean of the dataset is subtracted from each image. This removes the average brightness of the image (already done))
    ( 3. Feature Standardization  Set dimensions to have zero-mean and unit-variance (already done))
    """

    # scale image to from [0, 255] to [0, 1]
    image = image.astype(np.float32)
    return image / 255

def reshape_and_scale_single_image_back(image):
    """ Wrapper for utils.reshape_to_cv_format. """

    image, _ = utils.reshape_to_cv_format(image, True)
    return image

class NeuralNetClassifier(Classifier):
    """A wrapper class for the different neuralnets in nets"""
    
    def __init__(self, testData, name, description=""):               
        super(NeuralNetClassifier, self).__init__(-1, name, description, testData)
        
        self.testData.segment_test_data(Settings.NN_TESTDATA_SEGMENTS)
        self.testData.new_segmentation()
            
        self.x_train = None
        self.y_train = None
            
        self.net = None
        self.modelSaver = NeuralNetSaver(self.testData.get_root_path(), self.name, self.net, False, description) 


    def reshape_and_scale_single_image(self, image):
        image = utils.rescale_image_01(image)

        # reshape to (n, colorChannels, image_w, image_h)
        return image.reshape(1L, Settings.NN_CHANNELS, self.__imageSize[0], self.__imageSize[1])

    @property
    def imageSize(self):
        return self.__imageSize[0] * self.__imageSize[1]

    # quick and dirty hack (sorry)
    @imageSize.setter
    def imageSize(self, value):
        pass

    
    def load_data(self, dummy=False):
        """ 
        Loads the data before training which is a requirement of Lasagne / Nolearn. 

        Keyword arguments:
        dummy -- instead of loading actual images into memory small (1x1) dummy pixels are loaded into memory.
        """
              
        self.grayscale = Settings.NN_CHANNELS == 1L
        self.__imageSize = Settings.NN_INPUT_SHAPE    
            
        if dummy:
            print "Loading dummy data"
            self.x_train, self.y_train = self.testData.load_segment_dummy("train", self.__imageSize)
            return 

        x_train, y_train = self.testData.load_segment("train", grayscale=self.grayscale, size=self.__imageSize, transformation=utils.rescale_image_01)
        # convert responses to float 
        x_train = np.float32(x_train)
        y_train = np.int32(y_train)
        
        # reshape to (n, colorChannels, image_w, image_h)
        self.x_train = x_train.reshape(x_train.shape[0], Settings.NN_CHANNELS, self.__imageSize[0], self.__imageSize[1])
        self.y_train = y_train.reshape(y_train.shape[0], )

        if Settings.G_DETAILED_CONSOLE_OUTPUT:
            print "Data loaded.\n"
            print utils.get_table(["x_train Shape", "y_train Shape", "epochs"], 4, [self.x_train.shape, self.y_train.shape, Settings.NN_EPOCHS])

    def train_and_evaluate(self, save=True):
        if self.net is None:
            self.net = currentNet(self.testData.numberOfClasses, self.testData, self.modelSaver)

        if self.x_train is None or self.y_train is None:
            self.load_data(True)
        if Settings.G_DETAILED_CONSOLE_OUTPUT:
            print "\nTraining net {0}.".format(str(self.net))

        start = time.clock()      
        try:    
            self.net.fit(self.x_train, self.y_train)
        except Exception, e:
            logging.exception("Exception while fitting:")
            logging.debug("\nForce saving model.")
            self.modelSaver(self.net.net, self.net.net.train_history_, True)
        timeForFitting = time.clock() - start
        timePerEpoch = timeForFitting / Settings.NN_EPOCHS

        if Settings.G_DETAILED_CONSOLE_OUTPUT:
            print "\nFitting complete."

        if Settings.G_EVALUATION_DETAIL_HIGH:
            # plot loss history
            try:
                path = plot_train_history(self.net.net.train_history_, str(self.net), self.name, [("train_loss", "train loss"), ("valid_loss", "valid loss")], "loss", "Loss", ylimit=None, yscale=None, path=self.modelSaver.get_save_path_for_visualizations(), fileName="LossHistory")
                print "Saved loss training history in {0}.".format(path)
            except Exception, e:
                logging.exception("Could not save training history.")

            # plot accuracy
            try:
                path = plot_train_history(self.net.net.train_history_, str(self.net), self.name, [("valid_accuracy", "Accuracy")], "accuracy", "Accuracy", ylimit=None, yscale=None, path=self.modelSaver.get_save_path_for_visualizations(), fileName="AccuracyHistory")
                print "Saved accuracy training history in {0}.".format(path)
            except Exception, e:
                logging.exception("Could not save accuracy history.")

            
        # Test neural net
        self.tester = ModelTester(self, transformation=self.reshape_and_scale_single_image, size=self.__imageSize, transformationBack=reshape_and_scale_single_image_back)
        testClassifier = utils.radio_question("[?]", "Test the classifier?", None, ["Yes", "No"], [True, False])
        if testClassifier:            
            results = self.tester.test_classifier(["test"])
            testAccuracy = results["test"][0]
            testLoss = results["test"][1]
            print utils.get_table(["epochs", "valid accuracy", "valid loss", "training time", "time per epoch"], 2, [Settings.NN_EPOCHS, testAccuracy, testLoss, timeForFitting, timePerEpoch])

    def plot_training_history(self):
        """ Replot training history after training."""

        print "Plotting all epochs again."
        trainHistory = self.net.net.train_history_
        numberOfEpochs = len(trainHistory)
        th = TrainingHistory(self.name, str(self.net), None, self.modelSaver)
        for i in xrange(numberOfEpochs):
            th(self.net.net, trainHistory[:i+1])
            utils.show_progress(Settings.G_DETAILED_CONSOLE_OUTPUT, i+1, numberOfEpochs, "Plotting")
            
    def export_training_history_to_csv(self):     
        """ Export the training history to csv."""
           
        trainHistory = self.net.net.train_history_   
        numberOfEpochs = len(trainHistory)
        print "Exporting to csv. Number of epochs: {0}".format(numberOfEpochs)
        csv = [["epoch", "train_loss", "valid_loss", "valid accuracy", "duration"]]
        for epochInfo in trainHistory:
            csv.append([epochInfo["epoch"], epochInfo["train_loss"], epochInfo["valid_loss"], epochInfo["valid_accuracy"], epochInfo["dur"]])
            utils.show_progress(Settings.G_DETAILED_CONSOLE_OUTPUT, epochInfo["epoch"], numberOfEpochs, "Exporting")
        exporter = self.modelSaver.get_csv_exporter()
        exporter.export(csv, name="trainingHistory")
               
    def load_model(self, path, bestWeights):
        try:
            with open(path, "r") as f:
                self.net = pickle.load(f)
            self.net.set_network_specific_settings()
        except:
            logging.exception("Could not load model with pickle. Adjust currentNet if model can't be loaded. Model path: {0}".format(path))
            epoch = utils.value_question("[?]", "What was the number of the last epoch?", "i")
            self.net = currentNet(self.testData.numberOfClasses, self.testData, self.modelSaver)
            # pseudo restore train history
            trainHistory = []
            for i in xrange(epoch):
                trainHistory.append({"epoch":i, "train_loss":10, "valid_loss":10, "valid_accuracy":0, "dur":200})
            self.net.net.train_history_ = trainHistory
        print "Loaded Network: {0}".format(self.net)
        
        self.grayscale = Settings.NN_CHANNELS == 1L
        self.__imageSize = Settings.NN_INPUT_SHAPE
        # restore tester
        self.tester = ModelTester(self, transformation=self.reshape_and_scale_single_image, size=self.__imageSize, transformationBack=reshape_and_scale_single_image_back)

        # restore best weights if needed
        if bestWeights != None:
            self.net.net.load_params_from(bestWeights)

    def predict(self, images):
        """ Predict image."""
        if not self.trained:
            print "Classifier is not trained."
            continue_ = utils.radio_question("[?]", "The classifier might not be trained. Continue?", None, ["Yes", "No"], [True, False])
            if not continue_:    
                raise Exception("Classifier is not trained.")
            self.trained = True
        
        output = []
        for img in images:
            try:
                predictProbas = self.net.net.predict_proba(img)
            except:
                logging.exception("Could not predict. Image shape: {0}".format(img.shape))
                return []
            # prepare predictions
            # produce output in the form of [(class_, [predictions[0]])]
            predictionList = []
            for i in xrange(self.testData.numberOfClasses):		
                predictionList.append((self.testData.classes[i], [predictProbas[0][i]]))
            return sorted(predictionList, key=operator.itemgetter(1), reverse=True)

    def __str__(self):
        return self.name + " - " + str(self.net)