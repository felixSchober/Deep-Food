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


class net_cifar_cnn(object):
	"""
	Basic Neural Net for the cifar-10 dataset
	Input       3x32x32
	Conv2d      20x32x32 (5x5 filter)
	MaxPool     20x16x16
	Conv2d      20x16x16
	MaxPool     20x8x8
	Hidden      1000
	Output      X
	"""
	
	def set_network_specific_settings(self):
		Settings.NN_INPUT_SHAPE = (32, 32)
		Settings.NN_CHANNELS = 3L

		# Just for testing
		if 4 in Settings.NN_AUGMENTATION_PARAMS:
			del Settings.NN_AUGMENTATION_PARAMS[4]
		if 5 in Settings.NN_AUGMENTATION_PARAMS:
			del Settings.NN_AUGMENTATION_PARAMS[5]


	def __init__(self, outputShape):
		self.set_network_specific_settings()

		self.net =  NeuralNet(
			layers=[('input', layers.InputLayer),

					('conv2d1', layers.Conv2DLayer),
					('maxpool1', layers.MaxPool2DLayer),

					('conv2d2', layers.Conv2DLayer),
					('maxpool2', layers.MaxPool2DLayer),

					('dense', layers.DenseLayer),

					('output', layers.DenseLayer),
					],
			input_shape=(None, Settings.NN_CHANNELS, Settings.NN_INPUT_SHAPE[0], Settings.NN_INPUT_SHAPE[1]), # variable batch size, 3 color shape row shape

			conv2d1_num_filters=20, conv2d1_filter_size=(5, 5), conv2d1_stride=(1, 1), conv2d1_pad=(2, 2), conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
			maxpool1_pool_size=(2, 2),

			conv2d2_num_filters=20, conv2d2_filter_size=(5, 5), conv2d2_stride=(1, 1), conv2d2_pad=(2, 2), conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
			maxpool2_pool_size=(2, 2),

			dense_num_units=1000, dense_nonlinearity=lasagne.nonlinearities.rectify, 
			
			output_num_units=outputShape, 
			output_nonlinearity=lasagne.nonlinearities.softmax,
			
			# optimization method:
			update=nesterov_momentum,
			update_learning_rate=theano.shared(utils.to_float32(Settings.NN_START_LEARNING_RATE)),
			update_momentum=theano.shared(utils.to_float32(Settings.NN_START_MOMENTUM)),

			regression=False, # classification problem
			max_epochs=Settings.NN_EPOCHS,
			verbose=1,
			)

	def fit(self, x, y):
		return self.net.fit(x, y)

	def __str__(self):
		return "CNN_I-C5x5-P2x2-C5x5-P2x2-D1000-O"


class net_cifar_cnn_do(net_cifar_cnn):
	"""
	cifar-10 net but with dropout
	Input       3x32x32
	Conv2d      20x32x32 (5x5 filter)
	MaxPool     20x16x16
	Conv2d      20x16x16
	MaxPool     20x8x8
	Dropout     0.5
	Hidden      1000
	Output      X
	"""
	
	def __init__(self, outputShape, testData, modelSaver):
		super(net_cifar_cnn_do, self).__init__(outputShape)
		modelSaver.model = self
		self.net =  NeuralNet(
			layers=[('input', layers.InputLayer),

					('conv2d1', layers.Conv2DLayer),
					('maxpool1', layers.MaxPool2DLayer),

					('conv2d2', layers.Conv2DLayer),
					('maxpool2', layers.MaxPool2DLayer),
					('dropout1', layers.DropoutLayer),
					('dense', layers.DenseLayer),
					('dropout2', layers.DropoutLayer),
					('output', layers.DenseLayer),
					],
			input_shape=(None, Settings.NN_CHANNELS, Settings.NN_INPUT_SHAPE[0], Settings.NN_INPUT_SHAPE[1]), # variable batch size, 3 color shape row shape

			conv2d1_num_filters=20, conv2d1_filter_size=(5, 5), conv2d1_stride=(1, 1), conv2d1_pad=(2, 2), conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
			maxpool1_pool_size=(2, 2),

			conv2d2_num_filters=20, conv2d2_filter_size=(5, 5), conv2d2_stride=(1, 1), conv2d2_pad=(2, 2), conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
			maxpool2_pool_size=(2, 2),

			dropout1_p=0.5,

			dense_num_units=1000, dense_nonlinearity=lasagne.nonlinearities.rectify, 
			dropout2_p=0.5,
			output_num_units=outputShape, 
			output_nonlinearity=lasagne.nonlinearities.softmax,
			
			# optimization method:
			update=nesterov_momentum,
			update_learning_rate=theano.shared(utils.to_float32(Settings.NN_START_LEARNING_RATE)),
			update_momentum=theano.shared(utils.to_float32(Settings.NN_START_MOMENTUM)),

			#batch_iterator_train=AugmentingLazyBatchIterator(Settings.NN_BATCH_SIZE, testData, "train", False, newSegmentation=False, loadingSize=(120,120)),
			#batch_iterator_test=LazyBatchIterator(Settings.NN_BATCH_SIZE, testData, "valid", False, newSegmentation=False, loadingInputShape=Settings.NN_INPUT_SHAPE),
			#train_split=TrainSplit(eval_size=0.0), # we cross validate on our own

			regression=False, # classification problem
			on_epoch_finished=[
				AdjustVariable('update_learning_rate', start=Settings.NN_START_LEARNING_RATE, stop=0.0001),
				AdjustVariable('update_momentum', start=Settings.NN_START_MOMENTUM, stop=0.999),
				TrainingHistory("Cifar10-do1", str(self), [1], modelSaver),
				EarlyStopping(300),
				modelSaver,
				],
			max_epochs=Settings.NN_EPOCHS,
			verbose=1,
			)

	def __str__(self):
		return "CNN_I(DA,LR,MO)-C5x5-P2x2-C5x5-P2x2-D5-D1000-D5-O"

