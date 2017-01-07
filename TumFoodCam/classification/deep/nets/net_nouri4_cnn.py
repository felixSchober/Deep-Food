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


class net_nouri4_cnn(object):
	"""
	Net 4 used in Daniel Nouris nolearn tutorial http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
	This is basically the same as net3 but uses learning rate update and momentum update and no augmentation.
	Input       1x96x96
	Conv2d      32x94x94 (5x5 filter)
	MaxPool     32x47x47
	Conv2d      64x46x46
	MaxPool     64x23x23
	Conv2d      128x22x22
	MaxPool     128x11x11
	Hidden      500
	Hidden      500
	Output      X
	"""


	def set_network_specific_settings(self):
		Settings.NN_INPUT_SHAPE = (96, 96)
		Settings.NN_CHANNELS = 1L
		Settings.NN_START_LEARNING_RATE = 0.03
		Settings.NN_START_MOMENTUM = 0.9
		# grayscale so let's not use brightness or saturation changes (last two entries)
		if 4 in Settings.NN_AUGMENTATION_PARAMS:
			del Settings.NN_AUGMENTATION_PARAMS[4]
		if 5 in Settings.NN_AUGMENTATION_PARAMS:
			del Settings.NN_AUGMENTATION_PARAMS[5]


	def __init__(self, outputShape, modelSaver):        
		self.set_network_specific_settings()
		modelSaver.model = self
		self.net =  NeuralNet(
			layers=[
					('input', layers.InputLayer),
					('conv1', layers.Conv2DLayer),
					('pool1', layers.MaxPool2DLayer),
					('conv2', layers.Conv2DLayer),
					('pool2', layers.MaxPool2DLayer),
					('conv3', layers.Conv2DLayer),
					('pool3', layers.MaxPool2DLayer),
					('hidden4', layers.DenseLayer),
					('hidden5', layers.DenseLayer),
					('output', layers.DenseLayer),
					],

			input_shape=(None, Settings.NN_CHANNELS, Settings.NN_INPUT_SHAPE[0], Settings.NN_INPUT_SHAPE[1]), # variable batch size, 3 color shape row shape

			conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),

			conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),

			conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),

			hidden4_num_units=500, 
			hidden5_num_units=500,
			
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
				EarlyStopping(30),
				modelSaver,
				],
			max_epochs=Settings.NN_EPOCHS,
			verbose=1,
			)

	def fit(self, x, y):
		return self.net.fit(x, y)

	def __str__(self):
		return "CNN_I(LR,MO)-C3x3-P2x2-C2x2-P2x2-C2x2-P2x2-D500-D500-O"


