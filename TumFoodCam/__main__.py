import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import sys
import logging
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import data_io.settings as Settings
from misc import utils
from segmentation.image_segmentation import *
from classification.model_controller import ParameterTester, Testsuite, MODEL_CONTROLLER, ClassifierType
from data_io.model_IO import ModelLoader, remove_model
from misc.model_tester import ModelTester
from data_io.highscore_service import HighscoreService
from data_io.testdata import TestData, load_image
from classification.model import ModelType
import data_io.mail_service as MailServer
from data_io.settings_IO_service import SettingsService
from misc.color_space import ColorSpace


class TumFoodCam(object):
    """
    Main class for navigation and managing classification.
    """


    def __init__(self, mailReporting):
        self.__testData = None

        # Load settings
        self.settingsService = SettingsService()
        self.settingsService.load()

        # Manual Settings for testing override restored settings
        # Histogram Test
        #Settings.H_IMAGE_SIZE = 16384 # eq. 400x400
        #Settings.H_CROSS_VALIDATION_K = 1
        #Settings.H_IMAGE_SEGMENTS = 16
        #Settings.H_BINS = 150
        #Settings.H_TESTDATA_SEGMENTS = {"trainSVM": 0.8, "test": 0.2}
        ##Settings.H_MODEL_TYPE = ModelType.Sklearn
        #Settings.H_MODEL_TYPE = ModelType.KNearest
        #Settings.H_COLOR_SPACE = ColorSpace.HSV

        ## SIFT Test
        #Settings.E_NUMBER_OF_KEYPOINTS = 1152
        #Settings.E_RANDOM_SAMPLE = True
        #Settings.E_SCORE_WEIGHTING = True        
        #Settings.E_IMAGE_SIZE = 16384
        #Settings.E_MODEL_TYPE = ModelType.Sklearn
        #Settings.SI_CROSS_VALIDATION_K = 1
        #Settings.SI_TESTDATA_SEGMENTS = {"trainSVM": 0.8, "test": 0.2}
        #Settings.SI_BOW_DIMENSION = 2250

        # SURF Test
        #Settings.E_NUMBER_OF_KEYPOINTS = 1088
        #Settings.E_RANDOM_SAMPLE = False
        #Settings.E_SCORE_WEIGHTING = False        
        #Settings.E_IMAGE_SIZE = 262144
        #Settings.E_MODEL_TYPE = ModelType.Sklearn
        #Settings.SU_CROSS_VALIDATION_K = 1
        #Settings.SU_TESTDATA_SEGMENTS = {"trainSVM": 0.8, "test": 0.2}
        #Settings.SU_BOW_DIMENSION = 2500

        # ORB Test
        #Settings.O_CROSS_VALIDATION_K = 1
        #Settings.O_TESTDATA_SEGMENTS = {"trainSVM": 0.8, "test": 0.2}
        #Settings.O_BOW_DIMENSION = 2000
        #Settings.E_NUMBER_OF_KEYPOINTS = 1088
        #Settings.E_RANDOM_SAMPLE = True
        #Settings.E_SCORE_WEIGHTING = True        
        #Settings.E_IMAGE_SIZE = 262144
        #Settings.E_MODEL_TYPE = ModelType.Sklearn

        # LBP Test
        #Settings.L_SVM_PARAMS = dict(kernel_type=cv.SVM_POLY, svm_type=cv.SVM_C_SVC, C=10000, degree=2)
        #Settings.L_CROSS_VALIDATION_K = 1
        #Settings.L_TESTDATA_SEGMENTS = {"trainSVM": 0.8, "test": 0.2}
        #Settings.L_RADIUS = 8
        #Settings.L_NUMBER_OF_POINTS = 24 
        #Settings.L_EPS = 1e-7
        #Settings.L_METHOD = "uniform"
        #Settings.L_MODEL_TYPE = ModelType.OpenCV

        # CENSURE Test
        #Settings.E_NUMBER_OF_KEYPOINTS = 1088
        #Settings.E_RANDOM_SAMPLE = True
        #Settings.E_SCORE_WEIGHTING = False        
        #Settings.E_IMAGE_SIZE = 262144
        #Settings.E_MODEL_TYPE = ModelType.Sklearn
        #Settings.C_SVM_PARAMS = dict(kernel_type=cv.SVM_RBF, svm_type=cv.SVM_C_SVC, C=10000)
        #Settings.C_CROSS_VALIDATION_K = 1
        #Settings.C_TESTDATA_SEGMENTS = {"trainSVM": 0.8, "test": 0.2}
        #Settings.C_BOW_DIMENSION = 2500


        # NN Test
        #Settings.NN_EPOCHS = 1000
        
        #self.save_settings()

    def start_menue(self):
        """Starting point of program."""


        utils.clear_screen()
        options = ["create NEW classifier", "load EXISTING classifier", "test parameters", "show high scores", "segment image", "test data manipulation", "settings"]
        input = utils.menue("START MENUE:", options, True, False)

        if input == 1:
            self.create_classifier()
        elif input == 2:
            self.load_classifier()
        elif input == 3:
            self.test_parameters()
        elif input == 4:
            self.show_highscores()
        elif input == 5:
            self.segment_image()
        elif input == 6:
            self.test_data_menue()        
        elif input == 7:
            self.show_settings()
        elif input == 8:
            return

        self.start_menue()
        
    def create_classifier(self):
        """ Creates a model by using the ModelController. """

        modelId = MODEL_CONTROLLER.show_model_screen()
        if modelId == -1:
            return

        MODEL_CONTROLLER.set_model(modelId)
        
        if modelId == ClassifierType.VOTE:
            modelLoaded = MODEL_CONTROLLER.model.initialize()
            if not modelLoaded:
                MODEL_CONTROLLER.model = None
                self.start_menue()   
                return            

        MODEL_CONTROLLER.run_and_test(False)
        # continue to main model
        self.main_menue()
                 
    def load_classifier(self):
        utils.clear_screen()
        
        
        if not MODEL_CONTROLLER.load_model():
            self.start_menue()
        else:            
            self.main_menue()         
        
    def test_parameters(self):
        """ Main function to test one or more parameters of a classifier."""

        classifier = MODEL_CONTROLLER.show_model_screen()
        if classifier == -1:
            return
        name = utils.value_question("[?]", "Provide a name for the test run", "s")
        description = utils.value_question("[?]", "Provide a description for the test run", "s", "(optional)", True)

        # TODO: get parameter id
        ts = Testsuite(iterations=-1, random=False)
        ts.add_parameter(4, "Number of keypoints", range(32, 1182, 32))        
        tester = ParameterTester(name, description, classifier, ts)
        tester.start_testing()

    def test_data_menue(self):        
        utils.clear_screen()
        options = ["equalize test data image size", "add prefix to test data", "crop bounding boxes", "crop to square", "augment test data size", "cherry pick 'test data augmentation' iteration"]
        input = utils.menue("Test data menue", options, False, True)

        if input == 1:
            self.normalize_testdata()
        elif input == 2:
            self.rename_prefix()
        elif input == 3:
            self.crop_bounding_boxes()
        elif input == 4:
            self.crop_test_data_to_square()
        elif input == 5:
            self.augment_test_data()
        elif input == 6:
            self.augment_test_data(True)
        else:
            return
        self.test_data_menue()

    def normalize_testdata(self):
        """ Deprecated Menu: Resizes every image in a physical dataset to be the same."""

        size = utils.value_question("[?]", "Provide desired size (w * h)", "i", "250000 for 500x500 images")
        newName = utils.value_question("[?]", "Provide new name", "s", "Leave blank if you want to overwrite images", True)

        normDefaultData = utils.radio_question("[?]", "Use default dataset?", None, ["Yes", "No"], [True, False])
        datasetPath = ""
        if normDefaultData:
            datasetPath = Settings.G_TEST_DATA_PATH
        else:
            datasetPath = utils.value_question("[?]", "Provide path to test data", "s")

        data = TestData(datasetPath, 1)
        data.normalize_test_data(size, newName)

    def rename_prefix(self):
        """Menu for renaming images in a dataset so that datasets can be merged / combined without overwriting files by merging."""

        prefix = utils.value_question("[?]", "Provide prefix", "s", "Please do not use underscores (_).", True)

        normDefaultData = utils.radio_question("[?]", "Use default dataset?", None, ["Yes", "No"], [True, False])
        datasetPath = ""
        if normDefaultData:
            datasetPath = Settings.G_TEST_DATA_PATH
        else:
            datasetPath = utils.value_question("[?]", "Provide path to test data", "s")

        data = TestData(datasetPath, 1)
        data.add_prefix_to_test_data(prefix)

    def crop_bounding_boxes(self):
        """Menu for cropping the bounding boxes of datasets (Works with Food-100)."""

        boundingBoxFilename = utils.value_question("[?]", "Provide bounding box file name", "s", "Do not forget the file extension.", True)

        normDefaultData = utils.radio_question("[?]", "Use default dataset?", None, ["Yes", "No"], [True, False])
        datasetPath = ""
        if normDefaultData:
            datasetPath = Settings.G_TEST_DATA_PATH
        else:
            datasetPath = utils.value_question("[?]", "Provide path to test data", "s")

        data = TestData(datasetPath, 1)
        data.crop_bounding_boxes(boundingBoxFilename)

    def crop_test_data_to_square(self):
        """ Deprecated Menu: Crops all images in a dataset to squares."""

        rejectedName = utils.value_question("[?]", "Provide folder name for images that get rejected due to bad aspect ratio", "s", "Do not use the same name as your current data set.", False)

        normDefaultData = utils.radio_question("[?]", "Use default dataset?", None, ["Yes", "No"], [True, False])
        datasetPath = ""
        if normDefaultData:
            datasetPath = Settings.G_TEST_DATA_PATH
        else:
            datasetPath = utils.value_question("[?]", "Provide path to test data", "s")

        data = TestData(datasetPath, 1)
        data.crop_test_data_to_square(rejectedName)

    def augment_test_data(self, cherryPick=False):
        """ Deprecated Menu: Augments the test data."""
        
        iteration = -1
        equalizeAfter2ndIterationSize = -1 # no equalization
        if cherryPick:
            newName = ""
            iteration = utils.radio_question("[?]", "Pick iteration", None, [1, 2, 3], [1, 2, 3])        
        else:
            newName = utils.value_question("[?]", "Provide new name for data set", "s", "Do not use the same name as your current data set.", False)
            if utils.radio_question("[?]", "Equalize after second iteration?", None, ["Yes", "No"], [True, False]):
                equalizeAfter2ndIterationSize = utils.value_question("[?]", "Provide desired size (w * h)", "i", "250000 for 500x500 images")

        normDefaultData = utils.radio_question("[?]", "Use default dataset?", None, ["Yes", "No"], [True, False])
        datasetPath = ""
        if normDefaultData:
            datasetPath = Settings.G_TEST_DATA_PATH
        else:
            datasetPath = utils.value_question("[?]", "Provide path to test data", "s")

        data = TestData(datasetPath, 1)
        data.augment_test_data(newName, iteration, equalizeAfter2ndIterationSize)

    def show_settings(self):
        utils.clear_screen()

        # print current settings
        print "SETTINGS"
        print "DataSet:\t\t",self.show_default_dataset()
        print "High Eval Detail:\t",Settings.G_EVALUATION_DETAIL_HIGH
        print "Detailed output:\t",Settings.G_DETAILED_CONSOLE_OUTPUT
        print "Mail Notifications:\t",MailServer.are_mails_enabled()
        print "Auto report mailing:\t",Settings.G_AUTOMATIC_REPORT_MAILING
        

        options = ["change default data set", "change evaluation detail level", "change console output level", "SIFT Settings", "Histogram Settings", "SURF Settings", "Configure Mail Settings"]
        input = utils.menue("OPTIONS:", options, False, True)

        if input == 1:
            self.change_default_dataset()
            self.show_settings()
        elif input == 2:
            Settings.G_EVALUATION_DETAIL_HIGH = utils.radio_setting("Detail Level", Settings.G_EVALUATION_DETAIL_HIGH, ["High", "Low"], [True, False])
            self.save_settings()
            self.show_settings()
        elif input == 3:
            Settings.G_DETAILED_CONSOLE_OUTPUT = utils.radio_setting("Console Output Detail", Settings.G_DETAILED_CONSOLE_OUTPUT, ["High", "Low"], [True, False])
            self.save_settings()
            self.show_settings()
        elif input == 4:
            self.show_setting_SIFT()
            self.show_settings()
        elif input == 5:
            self.show_setting_histogram()
            self.show_settings()
        elif input == 6:
            self.show_setting_SURF()
            self.show_settings()
        elif input == 7:
            self.show_setting_Mail()
            self.show_settings()
        elif input == 8:
            self.start_menue()        
    
    def show_default_dataset(self):
        output = ""
        if Settings.G_TEST_DATA_PATH is None:
            output = "No data set provided."
        else:
            output = "Path to data set: {0}".format(Settings.G_TEST_DATA_PATH)
        if self.__testData is None:
            output += "\tDataset not loaded."
        else:
            output += "\tDataset loaded."
        return output

    def change_default_dataset(self):
        Settings.G_TEST_DATA_PATH = utils.value_setting("Test Data", "s", "Please provide the path directly to the image folder so that the folder only consists of folders of categories with images in them.")
        self.save_settings()

    def show_setting_histogram(self):
        utils.clear_screen()

        # print current settings
        print "Histogram SETTINGS"
        print "Cross Val. k:\t\t",Settings.H_CROSS_VALIDATION_K
        print "Test Segments:\t\t",Settings.H_TESTDATA_SEGMENTS
        print "Image Segments:\t\t",Settings.H_IMAGE_SEGMENTS
        print "Bins per Color\t\t",Settings.H_BINS
        print "Range:\t\t\t",Settings.H_COLOR_RANGE
        print "Color Space:\t\t",Settings.H_COLOR_SPACE
        print "SVM Props:\t\t",Settings.H_SVM_PARAMS

        options = ["change cross validation number", "change image segmentation", "change number of bins", "change color range", "change color space"]
        input = utils.menue("OPTIONS:", options, False, True)

        if input == 1:
            Settings.H_CROSS_VALIDATION_K = utils.value_setting("Cross validation runs", "i", "1 means no cross validation.")
            self.save_settings()
            self.show_setting_histogram()
        elif input == 2:
            Settings.H_IMAGE_SEGMENTS = utils.radio_setting("Number of Image Segments", Settings.H_IMAGE_SEGMENTS, ["1", "2", "4", "9", "16", "25"], [1, 2, 4, 9, 16, 25])
            self.save_settings()
            self.show_setting_histogram()
        elif input == 3:
            Settings.H_BINS = utils.value_setting("Number of histogram bins per color", "i", "default: 64 bins")
            self.save_settings()
            self.show_setting_histogram()
        elif input == 4:
            lower = utils.value_setting("Lower range for color values", "i", "default: 0")
            upper = utils.value_setting("Upper range for color values", "i", "default: 256")
            Settings.H_COLOR_RANGE = [lower, upper]
            self.save_settings()
            self.show_setting_histogram()
        elif input == 5:
            Settings.H_COLOR_SPACE = utils.radio_setting("Color Space", Settings.H_COLOR_SPACE, ["HSV", "RGB", "BGR"], [ColorSpace.HSV, ColorSpace.RGB, ColorSpace.BGR])
            self.save_settings()
            self.show_setting_histogram()
        else:
            self.show_settings()

    def show_setting_SIFT(self):
        utils.clear_screen()

        # print current settings
        print "SIFT SETTINGS"
        print "Cross Val. k:\t\t",Settings.SI_CROSS_VALIDATION_K
        print "Test Segments:\t\t",Settings.SI_TESTDATA_SEGMENTS
        print "SVM Props:\t\t",Settings.SI_SVM_PARAMS
        print "BOW Dimension:\t\t",Settings.SI_BOW_DIMENSION

        options = ["change cross validation number", "change BOW Dimension"]
        input = utils.menue("OPTIONS:", options, False, True)

        if input == 1:
            Settings.SI_CROSS_VALIDATION_K = utils.value_setting("Cross validation runs", "i", "1 means no cross validation.")
            self.save_settings()
            self.show_setting_SIFT()
        elif input == 2:
            Settings.SI_BOW_DIMENSION = utils.value_setting("Change BOW Dimension", "i", "default: 1000")
            self.save_settings()
            self.show_setting_SIFT()        
        else:
            self.show_settings()

    def show_setting_SURF(self):
        utils.clear_screen()

        # print current settings
        print "SURF SETTINGS"
        print "Cross Val. k:\t\t",Settings.SU_CROSS_VALIDATION_K
        print "Test Segments:\t\t",Settings.SU_TESTDATA_SEGMENTS
        print "SVM Props:\t\t",Settings.SU_SVM_PARAMS
        print "BOW Dimension:\t\t",Settings.SU_BOW_DIMENSION

        options = ["change cross validation number", "change BOW Dimension"]
        input = utils.menue("OPTIONS:", options, False, True)

        if input == 1:
            Settings.SU_CROSS_VALIDATION_K = utils.value_setting("Cross validation runs", "i", "1 means no cross validation.")
            self.save_settings()
            self.show_setting_SURF()
        elif input == 2:
            Settings.SU_BOW_DIMENSION = utils.value_setting("Change BOW Dimension", "i", "default: 1000")
            self.save_settings()
            self.show_setting_SURF()        
        else:
            self.show_settings()

    def show_setting_Mail(self):
        Settings.G_MAIL_FROM = utils.value_setting("Mail to send from", "s")
        Settings.G_MAIL_TO = utils.value_setting("Mail to send to", "s")
        Settings.G_MAIL_SERVER = utils.value_setting("SMTP Server URL", "s")
        Settings.G_MAIL_USER = utils.value_setting("Username", "s")
        Settings.G_MAIL_PASSWD = utils.value_setting("Password", "s", "Stored password won't be encryped!", False, True)        
        self.save_settings()

        if utils.radio_question("[?]", "Send test mail to " + Settings.G_MAIL_TO + "?", None, ["Yes", "No"], [True, False]):
            #from misc.MailService import MailService
            ms = MailServer.MailService()
            try:
                #ms.send_email("Setup complete", "Setup complete", [utils.get_data_path() + "mailTest.png"])
                Settings.G_AUTOMATIC_REPORT_MAILING = utils.radio_setting("Send automatic mail after the training of a classifier is complete? ", Settings.G_AUTOMATIC_REPORT_MAILING, ["Yes", "No"], [True, False])
                self.save_settings()
            except Exception, e:
                logging.exception("Could not send mail.")
                       
    def main_menue(self):
        """ Menu that is shown after classifier has been created or loaded."""

        # print params of loaded classifier
        modelUuid = MODEL_CONTROLLER.model.modelSaver.modelUuid
        ml = ModelLoader()
        try:
            params = ml.get_model_param(modelUuid)

            print "\n\nModel type:\t\t",params[0]
            print "Changed:\t\t",params[6]
            print "Name:\t\t\t",params[1]
            print "Model loss:\t\t",params[5]
            print "Model id:\t\t",modelUuid
            print "Model description:\t",params[2]
            print "Trained on:\t\t",params[3]
            print "\n"
        except:
            params = [""]

        options = ["mail reports", "predict random", "generate confusion matrix", "calculate accuracy results", "calculate accuracy results and insert score", "show high scores", "delete model", "predict image folder", "start webcam"]
        
        if params[0].startswith("NN") or params[0].startswith("CNN"):
            options.extend(["continue training", "plot training history", "export training history to csv"])
                 

        input = utils.menue("MAIN MENUE:", options)

        if input == 1:
            self.predict_random_image()
        elif input == 2:
            self.predict_random_image()
        elif input == 3:
            self.create_confusion_matrix()
        elif input == 4:
            self.calculate_results()
            raw_input("Press any key to continue.")
        elif input == 5:
            self.calculate_results(True)
            raw_input("Press any key to continue.")
        elif input == 6:
            self.show_highscores()
        elif input == 7:
            remove_model(modelUuid)
            print "Model removed."
            return self.start_menue()
        elif input == 8:
            self.predict_image_folder()
            raw_input("Prediction finished. Press any key to continue.")
        elif input == 9:
            self.predict_webcam()
            raw_input("Prediction finished. Press any key to continue.")
        elif input == 10 and (params[0].startswith("NN") or params[0].startswith("CNN")):
            # try to resume training
            MODEL_CONTROLLER.run_and_test(False)
        elif input == 11 and (params[0].startswith("NN") or params[0].startswith("CNN")):
            MODEL_CONTROLLER.model.plot_training_history()
            raw_input("Plotting complete. Press any key to continue.")
        elif input == 12 and (params[0].startswith("NN") or params[0].startswith("CNN")):
            MODEL_CONTROLLER.model.export_training_history_to_csv()
            raw_input("Exporting complete. Press any key to continue.")

        else:
            return self.start_menue()

        self.main_menue()

    def predict_random_image(self):
        """ Plots random predictions and a sliding window."""

        if MODEL_CONTROLLER.has_tester:
            tester = ModelTester(MODEL_CONTROLLER.model)
        else:
            tester = MODEL_CONTROLLER.tester
        tester.plot_random_predictions()

    def mail_reports(self):
        print "Does not work because Schlichter 2 and 4 do not seem to like sending mails."
      
    def predict_image_folder(self):
        """ Menu for predicting images in a given folder."""
        if MODEL_CONTROLLER.has_tester:
            tester = ModelTester(MODEL_CONTROLLER.model)
        else:
            tester = MODEL_CONTROLLER.tester
        path = utils.value_question("[...]", "Path to image folder", "s")
        tester.classify_image_folder(path)
        
    def predict_webcam(self):
        """ Menu for webcam (demo) prediction."""
        if MODEL_CONTROLLER.has_tester:
            tester = ModelTester(MODEL_CONTROLLER.model)
        else:
            tester = MODEL_CONTROLLER.tester
        tester.classify_webcam()
                  
    def create_confusion_matrix(self):
        """ Menu for confusion matrix creation."""
        if MODEL_CONTROLLER.has_tester:
            tester = ModelTester(MODEL_CONTROLLER.model)
        else:
            tester = MODEL_CONTROLLER.tester
        tester.plot_confusion_matrix()

    def calculate_results(self, insertScore=False):

        if not MODEL_CONTROLLER.has_tester:
            tester = ModelTester(MODEL_CONTROLLER.model)
        else:
            tester = MODEL_CONTROLLER.tester
        results = tester.test_classifier()
        print tester.format_results_string(results)

        if insertScore:        
            score = results[0][0]
            description = raw_input("Enter description for this result: ")
            scoreService = HighscoreService()
            scoreService = scoreService.load()
            position = scoreService.insert_score(score, description)
            scoreService.print_surrounding_highscores(position)
        
    def show_highscores(self):
        print "** HIGHSCORES **"
        scoreService = HighscoreService()
        scoreService = scoreService.load()
        scoreService.print_highscore()

    def segment_image(self):
        """ Menu for image segmentation."""
        choice = utils.menue("[Image Segmentation", ["Segment imgage", "Learn custom marker"], showBackToMain=True)
        if choice == 2:
            self.learn_marker()
            return
        elif choice == 3:
            return        

        im = load_image(utils.get_path_via_file_ui(), 1, 1)
        im = utils.equalize_image_size(im, 564000)
        seg = ObjectSegmentation()
        seg.process_image(im)
        im = seg.visulize_regions()
        cv.imwrite(utils.get_output_path() + "resultsMarker_regions.jpg", im)
            
        try:
            plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB), 'gray')
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()
            utils.save_plt_figure(plt, "image_segments")
        except:
            print "Could not plot / show image. Saving instead."
            utils.save_plt_figure(plt, "image_segments")

        colorSeg = ColorSegmentation()
        colorSeg.process_image(seg.get_meal_image())
        resultImg, images, titles = colorSeg.visulize_regions(seg)

        # if matplotlib does not work use cv
        for i in range(len(images)):
            cv.imwrite("C:/Users/felix/Desktop/Segmentation/results/resultsMarker_regionsArea{0}.jpg".format(titles[i]), images[i])        
            cv.imshow("Area {0}".format(titles[i]), images[i])
        cv.imshow("Result", resultImg)
        cv.imwrite("C:/Users/felix/Desktop/Segmentation/results/resultsMarker_resultsImage.jpg", resultImg) 
        cv.waitKey(0)

        # matplotlib 
        utils.display_images([resultImg], titles="Result", columns=1, rows=1)
        utils.display_images(images, titles=titles)   
        
    def learn_marker(self):
        seg = ObjectSegmentation()
        seg.learn_custom_marker()
    
    def save_settings(self):        
        self.settingsService.update_and_save_from_module()

# is the mail flag set?
mailReporting = "--mail" in sys.argv
print mailReporting    
program = TumFoodCam(mailReporting)
program.start_menue()
