import theano
import lasagne
from lasagne import layers, nonlinearities
from nolearn.lasagne import NeuralNet, TrainSplit
from lasagne.updates import nesterov_momentum
from classification.deep.learning_functions import AdjustVariable, EarlyStopping
from classification.deep.batch_iterators import AugmentingLazyBatchIterator, LazyBatchIterator
from classification.deep.training_history import TrainingHistory
from misc import utils
import data_io.settings as Settings
import numpy as np


class net_krizhevsky_cnn(object):
	"""
	Approximation of Krizhevsky's original ImageNet Deep Convolutional Neural Network presented in this paper from 2012:
	'Imagenet classification with deep convolutional neural networks'
	Input       3x224x224
	Conv2d      32x94x94 (5x5 filter)
	MaxPool     32x47x47
	Dropout     0.1
	Conv2d      64x46x46
	MaxPool     64x23x23
	Dropout     0.2
	Conv2d      128x22x22
	MaxPool     128x11x11
	Dropout     0.3
	Hidden      500
	Dropout     0.5
	Hidden      500
	Output      X
	"""


	def set_network_specific_settings(self):
		Settings.NN_INPUT_SHAPE = (224, 224)
		Settings.NN_CHANNELS = 3L
		Settings.NN_START_LEARNING_RATE = 0.01
		Settings.NN_START_MOMENTUM = 0.9
		if 4 in Settings.NN_AUGMENTATION_PARAMS:
			del Settings.NN_AUGMENTATION_PARAMS[4]
		if 5 in Settings.NN_AUGMENTATION_PARAMS:
			del Settings.NN_AUGMENTATION_PARAMS[5]


	def __init__(self, outputShape, testData, modelSaver):
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
					('conv4', layers.Conv2DLayer),
					('conv5', layers.Conv2DLayer),
					('pool3', layers.MaxPool2DLayer),
					('hidden6', layers.DenseLayer),
					('dropout1', layers.DropoutLayer),
					('hidden7', layers.DenseLayer),
					('dropout2', layers.DropoutLayer),
					('output', layers.DenseLayer),
					],


			input_shape=(None, Settings.NN_CHANNELS, Settings.NN_INPUT_SHAPE[0], Settings.NN_INPUT_SHAPE[1]), # variable batch size, 3 color shape row shape

			conv1_num_filters=96, conv1_filter_size=(11, 11), conv1_stride=(4, 4), 
			pool1_pool_size=(5, 5),
			

			conv2_num_filters=256, conv2_filter_size=(5, 5),
			pool2_pool_size=(3, 3),
			
			conv3_num_filters=384, conv3_filter_size=(3, 3), conv3_pad = (1,1),
			
			conv4_num_filters=384, conv4_filter_size=(3, 3), conv4_pad = (1,1),

			conv5_num_filters=256, conv5_filter_size=(3, 3), conv5_pad = (1,1),
			pool3_pool_size=(2, 2),

			hidden6_num_units=4096,
			dropout1_p=0.5,

			hidden7_num_units=4096,
			dropout2_p=0.5,

			
			output_num_units=outputShape, 
			output_nonlinearity=lasagne.nonlinearities.softmax,
			
			# optimization method:
			update=nesterov_momentum,
			update_learning_rate=theano.shared(utils.to_float32(Settings.NN_START_LEARNING_RATE)),
			update_momentum=theano.shared(utils.to_float32(Settings.NN_START_MOMENTUM)),

			batch_iterator_train=AugmentingLazyBatchIterator(Settings.NN_BATCH_SIZE, testData, "train", False, newSegmentation=False, loadingSize=(256,256)),
			batch_iterator_test=LazyBatchIterator(Settings.NN_BATCH_SIZE, testData, "valid", False, newSegmentation=False, loadingInputShape=Settings.NN_INPUT_SHAPE),
			train_split=TrainSplit(eval_size=0.0), # we cross validate on our own

			regression=False, # classification problem
			on_epoch_finished=[
				AdjustVariable('update_learning_rate', start=Settings.NN_START_LEARNING_RATE, stop=0.0001),
				AdjustVariable('update_momentum', start=Settings.NN_START_MOMENTUM, stop=0.999),
				TrainingHistory("Krizhevsky", str(self), [], modelSaver),
				EarlyStopping(150),
				modelSaver,
				],
			max_epochs=Settings.NN_EPOCHS,
			verbose=1,
			)

	def fit(self, x, y):
		return self.net.fit(x, y)

	def __str__(self):
		return "CNN_I(DA,LR,MO)-Krizhevsky-O"