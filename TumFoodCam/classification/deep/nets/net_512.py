import theano
import lasagne
from lasagne import layers, nonlinearities
from nolearn.lasagne import NeuralNet
from lasagne.updates import nesterov_momentum
from classification.deep.learning_functions import AdjustVariable, EarlyStopping
from classification.deep.batch_iterators import AugmentingBatchIterator
from classification.deep.training_history import TrainingHistory
from misc import utils
import data_io.settings as Settings
import numpy as np


class net_512(object):
    """
    Net 6 used in Daniel Nouris nolearn tutorial http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
    This is the same net as net5 but it contains dropout layers.
    Input       3x512x512
    Conv2d      32x254x254 (5x5 filter)
    MaxPool     32x127x127
    Conv2d      64x62x62
    MaxPool     64x31x31
    Conv2d      128x29x29
    MaxPool     128x14x14    
    Conv2d      192x12x12
    MaxPool     192x6x6
    Conv2d      256x4x4
    MaxPool     256x2x2
    Hidden      1024
    Maxout      512
    Hidden      1024
    Maxout      512
    Output      X
    """
    
    def set_network_specific_settings(self):
        Settings.NN_INPUT_SHAPE = (512, 512)
        Settings.NN_CHANNELS = 3L
        Settings.NN_START_LEARNING_RATE = 0.03
        Settings.NN_START_MOMENTUM = 0.9
        # grayscale so let's not use brightness or saturation changes (last two entries)
        del Settings.NN_AUGMENTATION_PARAMS[4]
        del Settings.NN_AUGMENTATION_PARAMS[5]


    def __init__(self, outputShape, modelSaver):
        self.set_network_specific_settings()
        modelSaver.model = self
        self.net =  NeuralNet(
                   layers=[
                        ('input', layers.InputLayer),
                        ('conv0', layers.Conv2DLayer),
                        ('pool0', layers.MaxPool2DLayer),
                        ('conv1', layers.Conv2DLayer),
                        ('pool1', layers.MaxPool2DLayer),
                        ('conv2', layers.Conv2DLayer),
                        ('pool2', layers.MaxPool2DLayer),
                        ('conv3', layers.Conv2DLayer),
                        ('pool3', layers.MaxPool2DLayer),
                        ('conv4', layers.Conv2DLayer),
                        ('pool4', layers.MaxPool2DLayer),
                        ('dropouthidden1', layers.DropoutLayer),
                        ('hidden1', layers.DenseLayer),
                        ('maxout1', layers.pool.FeaturePoolLayer),
                        ('dropouthidden2', layers.DropoutLayer),
                        ('hidden2', layers.DenseLayer),
                        ('maxout2', layers.pool.FeaturePoolLayer),
                        ('dropouthidden3', layers.DropoutLayer),
                        ('output', layers.DenseLayer),
                    ],



            input_shape=(None, Settings.NN_CHANNELS, Settings.NN_INPUT_SHAPE[0], Settings.NN_INPUT_SHAPE[1]), # variable batch size, 3 color shape row shape

            conv0_num_filters=32, conv0_filter_size=(5, 5), conv0_stride=(2, 2), pool0_pool_size=(2, 2), pool0_stride=(2, 2),

            conv1_num_filters=64, conv1_filter_size=(5, 5), conv1_stride=(2, 2), pool1_pool_size=(2, 2), pool1_stride=(2, 2),

            conv2_num_filters=128, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2), pool2_stride=(2, 2),

            conv3_num_filters=192, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2), pool3_stride=(2, 2),

            conv4_num_filters=256, conv4_filter_size=(3, 3), pool4_pool_size=(2, 2), pool4_stride=(2, 2),

            hidden1_num_units=1024,
            hidden2_num_units=1024,

            dropouthidden1_p=0.5,

            dropouthidden2_p=0.5,

            dropouthidden3_p=0.5,

            maxout1_pool_size=2,
            maxout2_pool_size=2,

            output_num_units=outputShape, 
            output_nonlinearity=lasagne.nonlinearities.softmax,
            
            # optimization method:
            update=nesterov_momentum,
            update_learning_rate=theano.shared(utils.to_float32(Settings.NN_START_LEARNING_RATE)),
            update_momentum=theano.shared(utils.to_float32(Settings.NN_START_MOMENTUM)),

            regression=False, # classification problem
            on_epoch_finished=[
                AdjustVariable('update_learning_rate', start=Settings.NN_START_LEARNING_RATE, stop=0.0001),
                AdjustVariable('update_momentum', start=Settings.NN_START_MOMENTUM, stop=0.999),
                TrainingHistory("?", str(self), [1], modelSaver),
                EarlyStopping(100),
                modelSaver,
                ],
            max_epochs=Settings.NN_EPOCHS,
            verbose=1,
            )

    def fit(self, x, y):
        return self.net.fit(x, y)

    def __str__(self):
        return "CNN_I(DA,LR,MO)-C5x5-P2x2-C5x5-P2x2-C3x3-P2x2-C3x3-P2x2-D1024-M500-D1024-M500-O"


