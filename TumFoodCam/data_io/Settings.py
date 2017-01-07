import cv2 as cv
from misc.color_space import ColorSpace
from classification.model import ModelType
from misc import utils
from random import uniform, randint
import numpy as np

G_TEST_DATA_PATH = ""
G_EVALUATION_DETAIL_HIGH = True
G_DETAILED_CONSOLE_OUTPUT = True
G_MAIL_REPORTS = False
G_MAIL_FROM = ""
G_MAIL_TO = ""
G_MAIL_SERVER = ""
G_MAIL_USER = ""
G_MAIL_PASSWD = ""
G_AUTOMATIC_REPORT_MAILING = False
G_COLOR_PALETTE = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725), (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), (0.5803921568627451, 0.403921568627451, 0.7411764705882353), (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), (0.8901960784313725, 0.4666666666666667, 0.7607843137254902), (0.4980392156862745, 0.4980392156862745, 0.4980392156862745), (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)] # Tableau-10 color palette

# PR - General image preprocessing settings
F_APPLY_NORMALIZATION = False
F_APPLY_ZNORMALIZATION = False
F_APPLY_ZCA_WHITENING = False
F_APPLY_CLAHE = False
F_APPLY_HISTOGRAM_EQ = False


# H - Histogram
H_IMAGE_SIZE = 160000 # eq. 400x400
H_CROSS_VALIDATION_K = 1
H_IMAGE_SEGMENTS = 4
H_BINS = 64
H_SCALE_HIST = False
H_COLOR_RANGE = [0, 256]
H_RESIZE_FACTOR = 1
H_TESTDATA_SEGMENTS = {"trainSVM": 0.8, "test": 0.2}
H_MODEL_TYPE = ModelType.OpenCV
H_SVM_PARAMS = dict(kernel_type=cv.SVM_POLY, svm_type=cv.SVM_C_SVC, C=10000, degree=2)
H_COLOR_SPACE = ColorSpace.HSV

# E - Settings for Edge classifiers
E_NUMBER_OF_KEYPOINTS = 1088
E_RANDOM_SAMPLE = True
E_SCORE_WEIGHTING = True
E_KEYPOINT_THRESHOLD = 0.55
E_KEYPOINT_LINEAR_THRESHOLD = np.linspace(0.9, 0, num=8)
E_IMAGE_SIZE = 16384
E_MODEL_TYPE = ModelType.Sklearn

# SI - SIFT
SI_CROSS_VALIDATION_K = 1
SI_RESIZE_FACTOR = 1
SI_TESTDATA_SEGMENTS = {"trainSVM": 0.8, "test": 0.2}
SI_SVM_PARAMS = dict(kernel_type=cv.SVM_RBF, svm_type=cv.SVM_C_SVC, C=8)
SI_BOW_DIMENSION = 800

# SU - SURF
SU_CROSS_VALIDATION_K = 1
SU_RESIZE_FACTOR = 1
SU_TESTDATA_SEGMENTS = {"trainSVM": 0.8, "test": 0.2}
SU_SVM_PARAMS = dict(kernel_type=cv.SVM_RBF, svm_type=cv.SVM_C_SVC, C=10000)
SU_BOW_DIMENSION = 800

# O - ORB
O_CROSS_VALIDATION_K = 1
O_TESTDATA_SEGMENTS = {"trainSVM": 0.8, "test": 0.2}
O_SVM_PARAMS = dict(kernel_type=cv.SVM_RBF, svm_type=cv.SVM_C_SVC, C=8)
O_BOW_DIMENSION = 800

# M - Multistep classifier
M_SVM_PARAMS = dict(kernel_type=cv.SVM_RBF, svm_type=cv.SVM_C_SVC, C=10000)

# L - Local binary pattern classifier
L_SVM_PARAMS = dict(kernel_type=cv.SVM_POLY, svm_type=cv.SVM_C_SVC, C=10000, degree=2)
L_CROSS_VALIDATION_K = 1
L_TESTDATA_SEGMENTS = {"trainSVM": 0.8, "test": 0.2}
L_RADIUS = 8
L_NUMBER_OF_POINTS = 24 
L_EPS = 1e-7
L_METHOD = "uniform"
L_MODEL_TYPE = ModelType.KNearest

# C - CENSURE Classifier
C_SVM_PARAMS = dict(kernel_type=cv.SVM_RBF, svm_type=cv.SVM_C_SVC, C=10000)
C_CROSS_VALIDATION_K = 1
C_TESTDATA_SEGMENTS = {"trainSVM": 0.8, "test": 0.2}
C_BOW_DIMENSION = 1000

# R - Random Keypoint Sampler
R_SVM_PARAMS = dict(kernel_type=cv.SVM_RBF, svm_type=cv.SVM_C_SVC, C=10000)
R_CROSS_VALIDATION_K = 1
R_TESTDATA_SEGMENTS = {"trainSVM": 0.8, "test": 0.2}
R_BOW_DIMENSION = 5

# K - NearestNeigbors
K_NUMBER_OF_NEIGHBORS_K = 10

# NN - Neural Net
NN_INPUT_SHAPE = (32, 32)
NN_TESTDATA_SEGMENTS = {"train": 0.8, "valid":0.1, "test": 0.1} 
NN_CROSS_VALIDATION_K = 1
NN_EPOCHS = 2000
NN_BATCH_SIZE = 128
NN_CHANNELS = 3L
NN_AUGMENT_DATA = True
NN_AUGMENT_CHANCE = np.linspace(0.6, 0, 5) # Chances for augmentation after n-th augmentation: [ 0.6 ,  0.45,  0.3 ,  0.15,  0.  ] -> maximum of four augmentations
NN_AUGMENTATION_PARAMS = {
                            0: (utils.translate_image_horizontal, (randint, -5, 5)), # translate_image_horizontal(image, translationFactor: [-5, 5]):
                            1: (utils.translate_image_vertical, (randint, -5, 5)), # translate_image_vertical(image, translationFactor: [-5, 5]):
                            2: (utils.flip_image, (randint, 0, 1)), # flip_image(image, direction: [0, 1]):                        
                            4: (utils.change_brightness_conv, (uniform, 0.25, 1.75)), # change_brightness_conv(image, value: [0.25 ~ 1.75]):
                            5: (utils.change_saturation_conv, (uniform, 0.25, 1.75)), # change_saturation_conv(image, value: [0.25 ~ 1.75]):
                            6: (utils.crop_image_to_shape, (randint, 0, 0), (randint, 0, 0), (lambda x, y: x, NN_INPUT_SHAPE, None)), # crop_image_to_shape(image, upperLeft_X: [0, ?*], upperLeft_Y: [0, ?*], shape: NN_INPUT_SHAPE): * has to be set by the iterator
                            7: (utils.rotate_image, (utils.choose_random_value, [90, 180, 270], True), (lambda x, y: x, False, None)), # rotate_image(image, angle: [90, 180, 270], crop: False):
                            8: (utils.rotate_image, (randint, -30, 30), (lambda x, y: x, True, None)) # rotate_image(image, angle: [-12, 12], crop: True):
                        }
NN_START_LEARNING_RATE = 0.001
NN_START_MOMENTUM = 0.9

