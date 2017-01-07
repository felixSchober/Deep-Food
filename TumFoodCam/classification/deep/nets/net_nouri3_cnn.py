import theano
import lasagne
from lasagne import layers, nonlinearities
from nolearn.lasagne import NeuralNet
from lasagne.updates import nesterov_momentum
from classification.deep.batch_iterators import AugmentingBatchIterator
from misc import utils
import data_io.settings as Settings
import numpy as np


class net_nouri3_cnn(object):
	"""
	Net 3 used in Daniel Nouris nolearn tutorial http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/#second-model-convolutions
	This is basically the same but uses data augmentation instead of no augmentation.
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
		# grayscale so let's not use brightness or saturation changes (last two entries)
		if 4 in Settings.NN_AUGMENTATION_PARAMS:
			del Settings.NN_AUGMENTATION_PARAMS[4]
		if 5 in Settings.NN_AUGMENTATION_PARAMS:
			del Settings.NN_AUGMENTATION_PARAMS[5]


	def __init__(self, outputShape):
		self.set_network_specific_settings()

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
			batch_iterator_train=AugmentingBatchIterator(batch_size=Settings.NN_BATCH_SIZE),
			max_epochs=Settings.NN_EPOCHS,
			verbose=1,
			)

	def fit(self, x, y):
		return self.net.fit(x, y)

	def __str__(self):
		return "CNN_I(DA)-C3x3-P2x2-C2x2-P2x2-C2x2-P2x2-D500-D500-O"


