import cv2 as cv
from misc.color_space import ColorSpace
from misc.utils import get_data_path
import pickle
import os
import data_io.settings as Settings


class SettingsService(object):
    """Class to load and save settings"""


    def __init__(self):

        # G - General
        self.G_TEST_DATA_PATH = ""
        self.G_EVALUATION_DETAIL_HIGH = True
        self.G_DETAILED_CONSOLE_OUTPUT = True
        self.G_MAIL_FROM = ""
        self.G_MAIL_SERVER = ""
        self.G_MAIL_USER = ""
        self.G_MAIL_PASSWD = ""
        self.G_MAIL_TO = ""

        # H - Histogram
        self.H_CROSS_VALIDATION_K = 5
        self.H_IMAGE_SEGMENTS = 4
        self.H_BINS = 64
        self.H_COLOR_RANGE = [0, 256]
        self.H_RESIZE_FACTOR = 1
        self.H_TESTDATA_SEGMENTS = {"trainSVM": 0.8, "test": 0.2}
        self.H_SVM_PARAMS = dict(kernel_type=cv.SVM_LINEAR, svm_type=cv.SVM_C_SVC, C=1)
        self.H_COLOR_SPACE = ColorSpace.HSV


        # SI - SIFT
        self.SI_CROSS_VALIDATION_K = 1
        self.SI_RESIZE_FACTOR = 1
        self.SI_TESTDATA_SEGMENTS = {"trainBoW": 0.2, "trainSVM": 0.6, "test": 0.2}
        self.SI_SVM_PARAMS = dict(kernel_type=cv.SVM_RBF, svm_type=cv.SVM_C_SVC, C=10000)
        self.SI_BOW_DIMENSION = 1000

        # SU - SURF
        self.SU_CROSS_VALIDATION_K = 1
        self.SU_RESIZE_FACTOR = 1
        self.SU_TESTDATA_SEGMENTS = {"trainBoW": 0.2, "trainSVM": 0.6, "test": 0.2}
        self.SU_SVM_PARAMS = dict(kernel_type=cv.SVM_RBF, svm_type=cv.SVM_C_SVC, C=10000)
        self.SU_BOW_DIMENSION = 1000

        # M - Multistep classifier
        self.M_SVM_PARAMS = dict(kernel_type=cv.SVM_LINEAR, svm_type=cv.SVM_C_SVC, C=1)
        

    def load(self):
        path = get_data_path() + "settings.dat"
        if not os.path.isfile(path):
            return self

        with open(path, "r") as f:
            loadedSettings = pickle.load(f)

        # update module values
        # G - General
        Settings.G_TEST_DATA_PATH = loadedSettings.G_TEST_DATA_PATH
        Settings.G_EVALUATION_DETAIL_HIGH = loadedSettings.G_EVALUATION_DETAIL_HIGH
        Settings.G_DETAILED_CONSOLE_OUTPUT = loadedSettings.G_DETAILED_CONSOLE_OUTPUT
        Settings.G_MAIL_FROM = loadedSettings.G_MAIL_FROM
        Settings.G_MAIL_TO = loadedSettings.G_MAIL_TO
        Settings.G_MAIL_SERVER = loadedSettings.G_MAIL_SERVER
        Settings.G_MAIL_USER = loadedSettings.G_MAIL_USER
        Settings.G_MAIL_PASSWD = loadedSettings.G_MAIL_PASSWD

        # H - Histogram
        #Settings.H_CROSS_VALIDATION_K = loadedSettings.H_CROSS_VALIDATION_K
        #Settings.H_IMAGE_SEGMENTS = loadedSettings.H_IMAGE_SEGMENTS
        #Settings.H_BINS = loadedSettings.H_BINS
        #Settings.H_COLOR_RANGE = loadedSettings.H_COLOR_RANGE
        #Settings.H_RESIZE_FACTOR = loadedSettings.H_RESIZE_FACTOR
        #Settings.H_TESTDATA_SEGMENTS = loadedSettings.H_TESTDATA_SEGMENTS
        #Settings.H_SVM_PARAMS = loadedSettings.H_SVM_PARAMS
        #Settings.H_COLOR_SPACE = loadedSettings.H_COLOR_SPACE


        ## SI - SIFT
        #Settings.SI_CROSS_VALIDATION_K = loadedSettings.SI_CROSS_VALIDATION_K
        #Settings.SI_RESIZE_FACTOR = loadedSettings.SI_RESIZE_FACTOR
        #Settings.SI_TESTDATA_SEGMENTS = loadedSettings.SI_TESTDATA_SEGMENTS
        #Settings.SI_SVM_PARAMS = loadedSettings.SI_SVM_PARAMS
        #Settings.SI_BOW_DIMENSION = loadedSettings.SI_BOW_DIMENSION

        ## SU - SURF
        #Settings.SU_CROSS_VALIDATION_K = loadedSettings.SU_CROSS_VALIDATION_K
        #Settings.SU_RESIZE_FACTOR = loadedSettings.SU_RESIZE_FACTOR
        #Settings.SU_TESTDATA_SEGMENTS = loadedSettings.SU_TESTDATA_SEGMENTS
        #Settings.SU_SVM_PARAMS = loadedSettings.SU_SVM_PARAMS
        #Settings.SU_BOW_DIMENSION = loadedSettings.SU_BOW_DIMENSION

        ## M - Multistep classifier
        #Settings.M_SVM_PARAMS = loadedSettings.M_SVM_PARAMS
            

    def update_and_save_from_module(self):
        # G - General
        self.G_TEST_DATA_PATH = Settings.G_TEST_DATA_PATH
        self.G_EVALUATION_DETAIL_HIGH = Settings.G_EVALUATION_DETAIL_HIGH
        self.G_DETAILED_CONSOLE_OUTPUT = Settings.G_DETAILED_CONSOLE_OUTPUT
        self.G_MAIL_FROM = Settings.G_MAIL_FROM
        self.G_MAIL_TO = Settings.G_MAIL_TO
        self.G_MAIL_SERVER = Settings.G_MAIL_SERVER
        self.G_MAIL_USER = Settings.G_MAIL_USER
        self.G_MAIL_PASSWD = Settings.G_MAIL_PASSWD

        # H - Histogram
        self.H_CROSS_VALIDATION_K = Settings.H_CROSS_VALIDATION_K
        self.H_IMAGE_SEGMENTS = Settings.H_IMAGE_SEGMENTS
        self.H_BINS = Settings.H_BINS
        self.H_COLOR_RANGE = Settings.H_COLOR_RANGE
        self.H_RESIZE_FACTOR = Settings.H_RESIZE_FACTOR
        self.H_TESTDATA_SEGMENTS = Settings.H_TESTDATA_SEGMENTS
        self.H_SVM_PARAMS = Settings.H_SVM_PARAMS
        self.H_COLOR_SPACE = Settings.H_COLOR_SPACE


        # SI - SIFT
        self.SI_CROSS_VALIDATION_K = Settings.SI_CROSS_VALIDATION_K
        self.SI_RESIZE_FACTOR = Settings.SI_RESIZE_FACTOR
        self.SI_TESTDATA_SEGMENTS = Settings.SI_TESTDATA_SEGMENTS
        self.SI_SVM_PARAMS = Settings.SI_SVM_PARAMS
        self.SI_BOW_DIMENSION = Settings.SI_BOW_DIMENSION

        # SU - SURF
        self.SU_CROSS_VALIDATION_K = Settings.SU_CROSS_VALIDATION_K
        self.SU_RESIZE_FACTOR = Settings.SU_RESIZE_FACTOR
        self.SU_TESTDATA_SEGMENTS = Settings.SU_TESTDATA_SEGMENTS
        self.SU_SVM_PARAMS = Settings.SU_SVM_PARAMS
        self.SU_BOW_DIMENSION = Settings.SU_BOW_DIMENSION

        # M - Multistep classifier
        self.M_SVM_PARAMS = Settings.M_SVM_PARAMS

        self.save()


    def save(self):
        path = get_data_path() + "settings.dat"        
        with open(path, "wb") as f:
            pickle.dump(self,f)
