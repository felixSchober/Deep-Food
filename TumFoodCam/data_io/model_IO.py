import numpy as np
import cPickle as pickle
import datetime
import csv
from misc import utils
from data_io.testdata import TestData
import logging
import data_io.settings as Settings

# Model Dictionary Layout: {modelUuid: (PARAMS)}
# Model Dictionary Params Layout:
#   [0] modelTypeId
#   [1] modelName
#   [2] modelDescription
#   [3] dataset Path
#   [4] model save path
#   [5] model loss
#   [6] date
#   [7] annotations

def get_model_dict():
    """ Loads and returns the model dictionary."""

    # load classifier dictionary
    path = utils.get_data_path() + "model_dictionary"

    # initialize if file doesn't exist
    modelDictionary = {}
    if utils.check_if_file_exists(path):  
        with open(path, "r") as f:
            modelDictionary = pickle.load(f) 
    return modelDictionary

def save_model_dict(modelDictionary):
    """ Saves the model dictionary to file."""

    with open(utils.get_data_path() + "model_dictionary", "wb") as f:
        pickle.dump(modelDictionary, f)

def update_model_dict(modelUuid, valueId, value):
    """ Updates and saves the model dictionary.

    Keyword arguments:
    modelUuid -- the uuid of the model to update
    valueId -- the index of the value to update
    value -- new value
    """

    modelDict = get_model_dict()
    modelParams = modelDict[modelUuid]

    newParams = []
    for i in xrange(len(modelParams)):
        if i != valueId:
            newParams.append(modelParams[i])
        else:
            newParams.append(value)

    modelDict[modelUuid] = tuple(newParams)
    save_model_dict(modelDict)


def remove_model(modelUuid):
    """ Removes the model with the modelUuid."""

    modelDict = get_model_dict()
    params = modelDict[modelUuid]

    # remove model dir
    utils.remove_dir(params[4])

    # remove model for dict
    del modelDict[modelUuid]
    save_model_dict(modelDict)


class ModelSaver(object):
    def __init__(self, datasetPath, name, description=""):
        """ Constructor for ModelSaver class.

        Keyword arguments:
        datasetPath -- Path to dataset of model
        name -- name of the model
        description -- description of the model
        """

        self.modelUuid = utils.get_uuid()
        self.modelName = name
        self.modelDescription = description
        self.datasetPath = datasetPath
        self.modelDictionary = get_model_dict()
        self.csvExporter = None
        

    def add_model_to_dict(self, modelPath, modelLoss, modelTypeId, annotations=None):  
        """
        Adds the model to the model dictionary.

        Keyword arguments:
        modelPath -- save root path of the model
        modelLoss -- the loss / error of the model.
        modelTypeId -- the model type id.
        annotations -- additional annotations like information about the saving method.
        """      

        now = datetime.datetime.now()
        dateString = "{0}.{1}.{2}".format(now.day, now.month, now.year)
        self.modelDictionary[self.modelUuid] = (modelTypeId, self.modelName, self.modelDescription, self.datasetPath, modelPath, modelLoss, dateString, annotations)
        save_model_dict(self.modelDictionary)

    def __call__(self, model, validLoss):
        """ Saves the model with the given validLoss."""

        path = self.get_save_path()
        model.save(path)
        self.add_model_to_dict(path, validLoss, str(model))

    def get_save_path(self):     
        """ Returns the current model root path."""    
        fileName = self.modelUuid
        path = utils.get_data_path() + fileName + "/"

        try:
            utils.create_dir_if_necessary(path)
        except:
            logging.exeption("Could not create dir to save classifier in. Saving in {0} instead.".format(utils.get_data_path()))
            path = utils.get_data_path()
        return path

    def get_save_path_for_visualizations(self):
        """ Returns the path for visualizations for the current model."""

        path = self.get_save_path() + "visualizations/"
        utils.create_dir_if_necessary(path)
        return path

    def get_csv_exporter(self):
        """ Returns the csv exporter for this model."""

        if self.csvExporter is None:
            self.csvExporter = CsvExporter(self)
        return self.csvExporter
    

class NeuralNetSaver(ModelSaver):
    """ A class to save NeuralNet instances."""
     
    def __init__(self, datasetPath, name, model, pickleModel=False, description=""):     
        """
        Instantiates a new NeuralNetSaver.

        Keyword arguments:
        datasetPath -- Path for the dataset
        name -- Name of the Neural Net
        model -- Neural Net
        pickleModel -- Pickle whole model or just save the weights
        description -- description of the net
        """   

        self.pickleModel = pickleModel
        self.model = model
        super(NeuralNetSaver, self).__init__(datasetPath, name, description)        
        self.bestLoss = np.inf
        

    def __call__(self, nn, trainHistory, force=False):   
        """ 
        Saves the model.

        Keyword arguments:
        nn -- neural net model
        trainHistory -- the training history of the neural net generated by nolearn
        force -- save model regardless of current loss.
        """
         
        path = self.get_save_path()
        validLoss = trainHistory[-1]["valid_loss"]
        currentEpoch = trainHistory[-1]["epoch"]
               
        # if pickleModel is true -> always save the model regardless of achieved accuracy as a pickeld class
        # save the model every 20th epoch
        if self.pickleModel or force or currentEpoch % 20 == 0:
            # pickle only model to prevent the pickelation (awesome word) of the test data
            with open(path + "model", "wb") as f:
                pickle.dump(self.model, f, -1)
            self.add_model_to_dict(path, validLoss, str(self.model))
        
        # save only params if the loss is better than previously reported            
        if self.bestLoss > validLoss:
            nn.save_params_to(path + "best_weights")
            self.add_model_to_dict(path, validLoss, str(self.model), annotations="nn_weights")


class SvmModelSaver(ModelSaver):
    """ A class to save SVM-like models."""

    def __init__(self, datasetPath, name, description=""):
        super(SvmModelSaver, self).__init__(datasetPath, name, description)

    def __call__(self, model, validLoss):
        """ Saves the model."""

        path = self.get_save_path()

        # save SVMs separately because OpenCV provides it's own method to save an SVM
        svmPath = path + "svms/"
        utils.create_dir_if_necessary(svmPath) 
        for svmName in model.svms:
             model.svms[svmName].save(svmPath)
             
        # pickle self   
        with open(path + "model", "wb") as f:
            pickle.dump(model,f)
        self.add_model_to_dict(path, validLoss, str(model))
        

class ModelLoader(object):
    """ Class to load a model."""

    def __init__(self):
        self.modelDictionary = get_model_dict()


    def load_model(self, modelUuid, manualPath=None):
        """ 
        Load model from either model dictionary or manually.
        
        Keyword arguments:
        modelUuid -- uuid of the model. If None the method will try to load the model using the manualPath.
        manualPath -- if model shall be loaded manually this is the root path of the model directory

        Returns:
        model
        """

        # manual mode. Load model that is not part of the model dictionary.
        if modelUuid is None:
            classifier = None
            try:
                with open(manualPath + "model", "rb") as f:
                    classifier = pickle.load(f)
            except:
                logging.exception("Could not load model manually")
                return None
            return classifier
            
        else:
            # Load model from model dictionary
            modelParams = self.get_model_param(modelUuid)
            modelSavePath = modelParams[4]
            modelTypeId = modelParams[0]            
            testdata = TestData(modelParams[3], 1, True)

        if not self.does_model_exist(modelUuid):
            raise AttributeError("Model with uuid {0} was not found in model dictionary.".format(modelUuid))
                
        if modelTypeId == "SIFT":
            from classification.local_features.sift import SIFTClassifier
            from classification.model import ModelType
            model = SIFTClassifier(testdata, Settings.E_MODEL_TYPE)
            model = model.load(modelSavePath)

        elif modelTypeId == "SURF":
            from classification.local_features.surf import SURFClassifier
            from classification.model import ModelType
            model = SURFClassifier(testdata, Settings.E_MODEL_TYPE)
            return model.load(modelSavePath)
        elif modelTypeId == "HIST":
            from classification.global_features.histogram import HistogramClassifier
            from classification.model import ModelType
            model = HistogramClassifier(testdata, Settings.E_MODEL_TYPE)
            return model.load(modelSavePath)

        if modelTypeId.startswith("mCL"):
            from classification.late_fusion import MajorityClassifier
            model = MajorityClassifier(testdata)
            try:
                with open(modelSavePath + "model", "r") as f:
                    model = pickle.load(f)
            except:
                logging.exception("Could not load majority classifier.")
                return None
            return model

        # NNs or CNNs
        if modelTypeId.startswith("NN") or modelTypeId.startswith("CNN"):            
            from classification.deep.neural_net import *                 
            # load testdata because we need the output shape       
            modelWrapper = NeuralNetClassifier(testdata, modelParams[3])

            # search for best weights
            if not utils.check_if_file_exists(modelSavePath + "model"):     
                print "[!] Model file {0} was not found.".format(modelSavePath + "model")      
                
                continue_ = utils.radio_question("[?]", "It might be possible to restore the model using the weights file. Continue?", None, ["Yes", "No"], [True, False])
                if not continue_:     
                    delete = utils.radio_question("[?]", "Delete model?", None, ["Yes", "No"], [True, False])
                    if delete:
                        remove_model(modelUuid)
                    raise Exception("Model file does not exist.")

            # try to restore best weights if more recent
            bestWeights = None
            if modelParams[7] == "nn_weights" and utils.check_if_file_exists(modelSavePath + "best_weights"):
                bestWeights = modelSavePath + "best_weights"

            modelWrapper.load_model(modelSavePath + "model", bestWeights)
            # restore params
            modelWrapper.modelSaver.bestLoss = modelParams[5]
            modelWrapper.modelSaver.modelDescription = modelParams[2]
            modelWrapper.modelSaver.modelUuid = modelUuid
            return modelWrapper    
        if modelTypeId is None or modelTypeId == "None":
            print "There was a problem loading this model {0}. The save file might be corrupted. Model Dictionary {1}".format(modelTypeId, modelParams)
            if utils.radio_question("[?]", "Repair model with new model type ID?", None, ["Yes", "No"], [True, False]):
                modelTypeId = utils.value_question("[?]", "Model ID", "s")
                update_model_dict(modelUuid, 0, modelTypeId)
                print "Model Id changed. Restart application and try again."
                raw_input("Press any key to continue.")
                import sys
                sys.exit()
            raise Exception("Could not repair model.")
                
                   
        else:
            raise Exception("Model {0} is not supported yet.".format(modelTypeId))

    def get_model_options(self):
        """ 
        Reads the model options from the model dictionary.

        returns two lists:
            [0] List: contains the uuids to load the models
            [1] List: contains the attributes describing the model
        """

        modelUuids = []
        modelParams = []
        for modelUuid in self.modelDictionary:
            modelUuids.append(modelUuid)
            modelParam = self.modelDictionary[modelUuid]
            modelParams.append("{0} - [{1}] {2} - Loss: {3}".format(modelParam[6], modelParam[0], modelParam[1], modelParam[5]))
        return (modelUuids, modelParams)

    def get_model_param(self, modelUuid):
        """ Returns the model parameters for the given uuid."""

        return self.modelDictionary[modelUuid]

    def does_model_exist(self, modelUuid):
        """ Returns True or False if a model with a given uuid exists."""

        return modelUuid in self.modelDictionary

    def show_loading_screen(self):
        """ 
        Loads a model by letting the user choose from a loading screen.

        Returns: 
        model
        """

        modelUuids, modelParams = self.get_model_options()
        if not modelUuids:
            print "No models available :("
            raw_input("Press any key to continue")
            return None
        else:
            modelParams.insert(0, "Load manually")
            modelUuids.insert(0, "manual")
            modelUuid = utils.radio_question("[Load Model]", "Model Loader", None, modelParams, modelUuids)
            model = None

            if modelUuid == "manual":
                path = utils.value_question("[...]", "Provide the path to the model directory", "s")
                if not utils.check_if_dir_exists(path):
                    raise Exception("Could not find path {0}.".format(path))
                try:
                    model = self.load_model(None, path)
                    return model
                except Exception, e:
                    logging.exception("Could not load model with Path {0}.".format(path))   
                    raw_input("Press any key to continue")
                    return None

            try:
                model = self.load_model(modelUuid)
                print "Model {0} was loaded successfully.".format(model.modelSaver.modelUuid)
                return model    
                
            except Exception, e:
                logging.exception("Could not load model with Uuid {0}. Restart application and try again.".format(modelUuid))                
                delete = utils.radio_question("[?]", "Delete model?", None, ["Yes", "No"], [True, False])
                if delete:
                    remove_model(modelUuid)
                raw_input("Press any key to continue")
                return None           


class CsvExporter(object):
    """ Simple csv exporter that saves reports into the root path of the model."""

    def __init__(self, modelSaver):
        self.modelSaver = modelSaver

    def export(self, data, name=None):
        """ 
        Exports the given data.

        Keyword arguments:
        data -- list of a list of the data. rows[cols[]]
        name -- optional name of the file. If None a uuid is generated."
        """

        if name is None:
            name = utils.get_uuid()
        path = self.modelSaver.get_save_path() + name + ".csv"
        with open(path, "wb") as f:
            writer = csv.writer(f, dialect="excel")
            writer.writerows(data)

        # return the path
        return path







