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


class net_nouri6_cnn(object):
	"""
	Net 6 used in Daniel Nouris nolearn tutorial http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
	This is the same net as net5 but it contains dropout layers.
	Input       1x96x96
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
		Settings.NN_INPUT_SHAPE = (96, 96)
		Settings.NN_CHANNELS = 1L
		Settings.NN_START_LEARNING_RATE = 0.03
		Settings.NN_START_MOMENTUM = 0.9
		# grayscale so let's not use brightness or saturation changes (last two entries)
		if 4 in Settings.NN_AUGMENTATION_PARAMS:
			del Settings.NN_AUGMENTATION_PARAMS[4]
		if 5 in Settings.NN_AUGMENTATION_PARAMS:
			del Settings.NN_AUGMENTATION_PARAMS[5]


	def __init__(self, outputShape, testData, modelSaver):
		self.set_network_specific_settings()
		modelSaver.model = self
		self.ini_net(outputShape, testData, modelSaver)

	def ini_net(self, outputShape, testData, modelSaver):
		self.net =  NeuralNet(
			layers=[
					('input', layers.InputLayer),
					('conv1', layers.Conv2DLayer),
					('pool1', layers.MaxPool2DLayer),
					('dropout1', layers.DropoutLayer),
					('conv2', layers.Conv2DLayer),
					('pool2', layers.MaxPool2DLayer),
					('dropout2', layers.DropoutLayer),
					('conv3', layers.Conv2DLayer),
					('pool3', layers.MaxPool2DLayer),
					('dropout3', layers.DropoutLayer),
					('hidden4', layers.DenseLayer),
					('dropout4', layers.DropoutLayer),
					('hidden5', layers.DenseLayer),
					('output', layers.DenseLayer),
					],


			input_shape=(None, Settings.NN_CHANNELS, Settings.NN_INPUT_SHAPE[0], Settings.NN_INPUT_SHAPE[1]), # variable batch size, 3 color shape row shape

			conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
			dropout1_p=0.1,

			conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
			dropout2_p=0.2,

			conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
			dropout3_p=0.3,

			hidden4_num_units=500,
			dropout4_p=0.5,

			hidden5_num_units=500,

			
			output_num_units=outputShape, 
			output_nonlinearity=lasagne.nonlinearities.softmax,
			
			# optimization method:
			update=nesterov_momentum,
			update_learning_rate=theano.shared(utils.to_float32(Settings.NN_START_LEARNING_RATE)),
			update_momentum=theano.shared(utils.to_float32(Settings.NN_START_MOMENTUM)),

			batch_iterator_train=AugmentingLazyBatchIterator(Settings.NN_BATCH_SIZE, testData, "train", False, newSegmentation=False, loadingSize=(120,120)),
			batch_iterator_test=LazyBatchIterator(Settings.NN_BATCH_SIZE, testData, "valid", False, newSegmentation=False, loadingInputShape=Settings.NN_INPUT_SHAPE),
			train_split=TrainSplit(eval_size=0.0), # we cross validate on our own

			regression=False, # classification problem
			on_epoch_finished=[
				AdjustVariable('update_learning_rate', start=Settings.NN_START_LEARNING_RATE, stop=0.0001),
				AdjustVariable('update_momentum', start=Settings.NN_START_MOMENTUM, stop=0.999),
				TrainingHistory("?", str(self), [1], modelSaver),
				EarlyStopping(150),
				modelSaver,
				],
			max_epochs=Settings.NN_EPOCHS,
			verbose=1,
			)


	def fit(self, x, y):
		return self.net.fit(x, y)

	def __str__(self):
		return "CNN_I(DA,LR,MO)-C3x3-P2x2-D1-C2x2-P2x2-D2-C2x2-P2x2-D3-D500-D5-D500-O"


class net_nouri6_cnn_do(net_nouri6_cnn):
	"""
	Same as Net 6 but with a higher dropout rate to support better generalization
	Input       1x96x96
	Conv2d      32x94x94 (5x5 filter)
	MaxPool     32x47x47
	Dropout     0.2
	Conv2d      64x46x46
	MaxPool     64x23x23
	Dropout     0.4
	Conv2d      128x22x22
	MaxPool     128x11x11
	Dropout     0.6
	Hidden      500
	Dropout     0.6
	Hidden      500
	Output      X
	"""


	def __init__(self, outputShape, testData, modelSaver):
		super(net_nouri6_cnn_do, self).__init__(outputShape, testData, modelSaver)


	def ini_net(self, outputShape, testData, modelSaver):
		self.net =  NeuralNet(
				layers=[
						('input', layers.InputLayer),
						('conv1', layers.Conv2DLayer),
						('pool1', layers.MaxPool2DLayer),
						('dropout1', layers.DropoutLayer),
						('conv2', layers.Conv2DLayer),
						('pool2', layers.MaxPool2DLayer),
						('dropout2', layers.DropoutLayer),
						('conv3', layers.Conv2DLayer),
						('pool3', layers.MaxPool2DLayer),
						('dropout3', layers.DropoutLayer),
						('hidden4', layers.DenseLayer),
						('dropout4', layers.DropoutLayer),
						('hidden5', layers.DenseLayer),
						('output', layers.DenseLayer),
						],


				input_shape=(None, Settings.NN_CHANNELS, Settings.NN_INPUT_SHAPE[0], Settings.NN_INPUT_SHAPE[1]), # variable batch size, 3 color shape row shape

				conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
				dropout1_p=0.2,

				conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
				dropout2_p=0.4,

				conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
				dropout3_p=0.6,

				hidden4_num_units=500,
				dropout4_p=0.6,

				hidden5_num_units=500,

			
				output_num_units=outputShape, 
				output_nonlinearity=lasagne.nonlinearities.softmax,
			
				# optimization method:
				update=nesterov_momentum,
				update_learning_rate=theano.shared(utils.to_float32(Settings.NN_START_LEARNING_RATE)),
				update_momentum=theano.shared(utils.to_float32(Settings.NN_START_MOMENTUM)),

				batch_iterator_train=AugmentingLazyBatchIterator(Settings.NN_BATCH_SIZE, testData, "train", False, newSegmentation=False, loadingSize=(120,120)),
				batch_iterator_test=LazyBatchIterator(Settings.NN_BATCH_SIZE, testData, "valid", False, newSegmentation=False, loadingInputShape=Settings.NN_INPUT_SHAPE),
				train_split=TrainSplit(eval_size=0.0), # we cross validate on our own

				regression=False, # classification problem
				on_epoch_finished=[
					AdjustVariable('update_learning_rate', start=Settings.NN_START_LEARNING_RATE, stop=0.0001),
					AdjustVariable('update_momentum', start=Settings.NN_START_MOMENTUM, stop=0.999),
					TrainingHistory("?", str(self), [1], modelSaver),
					EarlyStopping(150),
					modelSaver,
					],
				max_epochs=Settings.NN_EPOCHS,
				verbose=1,
				)

	def __str__(self):
		return "CNN_I(DA,LR,MO)-C3x3-P2x2-D2-C2x2-P2x2-D4-C2x2-P2x2-D6-D500-D6-D500-O"


class net_nouri6_cnn_do_color(net_nouri6_cnn):
	"""
	Same as Net 6 but with a higher dropout, smaller images and color
	Input       3x64x64
	Conv2d      32x50x50 (5x5 filter)
	MaxPool     32x25x25
	Dropout     0.3
	Conv2d      64x25x25
	MaxPool     64x23x23
	Dropout     0.4
	Conv2d      128x22x22
	MaxPool     128x11x11
	Dropout     0.6
	Hidden      500
	Dropout     0.6
	Hidden      500
	Output      X
	"""


	def set_network_specific_settings(self):
		Settings.NN_INPUT_SHAPE = (48, 48)
		Settings.NN_CHANNELS = 3L

	def __init__(self, outputShape, testData, modelSaver):
		super(net_nouri6_cnn_do_color, self).__init__(outputShape, testData, modelSaver)

	def ini_net(self, outputShape, testData, modelSaver):
		self.net =  NeuralNet(
				layers=[
						('input', layers.InputLayer),
						('conv1', layers.Conv2DLayer),
						('pool1', layers.MaxPool2DLayer),
						('dropout1', layers.DropoutLayer),
						('conv2', layers.Conv2DLayer),
						('pool2', layers.MaxPool2DLayer),
						('dropout2', layers.DropoutLayer),
						('conv3', layers.Conv2DLayer),
						('pool3', layers.MaxPool2DLayer),
						('dropout3', layers.DropoutLayer),
						('hidden4', layers.DenseLayer),
						('dropout4', layers.DropoutLayer),
						('hidden5', layers.DenseLayer),
						('output', layers.DenseLayer),
						],


				input_shape=(None, Settings.NN_CHANNELS, Settings.NN_INPUT_SHAPE[0], Settings.NN_INPUT_SHAPE[1]), # variable batch size, 3 color shape row shape

				conv1_num_filters=32, conv1_filter_size=(5, 5), conv1_stride=(1, 1), conv1_pad=(1, 1), conv1_nonlinearity=lasagne.nonlinearities.rectify,
				pool1_pool_size=(2, 2),
				dropout1_p=0.2,

				conv2_num_filters=64, conv2_filter_size=(5, 5), conv2_stride=(1, 1), conv2_pad=(1, 1), conv2_nonlinearity=lasagne.nonlinearities.rectify,
				pool2_pool_size=(2, 2),
				dropout2_p=0.4,

				conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2), conv3_nonlinearity=lasagne.nonlinearities.rectify,
				dropout3_p=0.5,

				hidden4_num_units=512, hidden4_nonlinearity=lasagne.nonlinearities.rectify,
				dropout4_p=0.6,

				hidden5_num_units=512, hidden5_nonlinearity=lasagne.nonlinearities.rectify,

			
				output_num_units=outputShape, 
				output_nonlinearity=lasagne.nonlinearities.softmax,
			
				# optimization method:
				update=nesterov_momentum,
				update_learning_rate=theano.shared(utils.to_float32(Settings.NN_START_LEARNING_RATE)),
				update_momentum=theano.shared(utils.to_float32(Settings.NN_START_MOMENTUM)),

				batch_iterator_train=AugmentingLazyBatchIterator(Settings.NN_BATCH_SIZE, testData, "train", False, newSegmentation=False, loadingSize=(150,150)),
				batch_iterator_test=LazyBatchIterator(Settings.NN_BATCH_SIZE, testData, "valid", False, newSegmentation=False, loadingInputShape=Settings.NN_INPUT_SHAPE),
				train_split=TrainSplit(eval_size=0.0), # we cross validate on our own

				regression=False, # classification problem
				on_epoch_finished=[
					AdjustVariable('update_learning_rate', start=Settings.NN_START_LEARNING_RATE),
					AdjustVariable('update_momentum', start=Settings.NN_START_MOMENTUM),
					TrainingHistory("?", str(self), [1], modelSaver),
					EarlyStopping(350),
					modelSaver,
					],
				max_epochs=Settings.NN_EPOCHS,
				verbose=1,
				)

	def __str__(self):
		return "CNN_I(DA,LR,MO)-C5x5-P2x2-D2-C5x5-P2x2-D4-C2x2-P2x2-D6-D512-D6-D512-O"


class net_nouri6_cnn_do2_color(net_nouri6_cnn):
	"""
	Same as Net 6 color but with a higher dropout, smaller images and color
	Input       3x64x64
	Conv2d      32x50x50 (5x5 filter)
	MaxPool     32x25x25
	Dropout     0.3
	Conv2d      64x25x25
	MaxPool     64x23x23
	Dropout     0.4
	Conv2d      128x22x22
	MaxPool     128x11x11
	Dropout     0.6
	Hidden      500
	Dropout     0.6
	Hidden      500
	Output      X
	"""


	def set_network_specific_settings(self):
		Settings.NN_INPUT_SHAPE = (48, 48)
		Settings.NN_CHANNELS = 3L
		#Settings.NN_START_LEARNING_RATE = 0.03
		#Settings.NN_START_MOMENTUM = 0.9
		#Settings.NN_AUGMENT_CHANCE = np.linspace(0.65, 0, 5)
		#Settings.NN_TESTDATA_SEGMENTS = {"train": 0.9, "test": 0.1}
		if 4 in Settings.NN_AUGMENTATION_PARAMS:
			del Settings.NN_AUGMENTATION_PARAMS[4]
		if 5 in Settings.NN_AUGMENTATION_PARAMS:
			del Settings.NN_AUGMENTATION_PARAMS[5]


	def __init__(self, outputShape, testData, modelSaver):
		super(net_nouri6_cnn_do2_color, self).__init__(outputShape, testData, modelSaver)

	def ini_net(self, outputShape, testData, modelSaver):
		self.net =  NeuralNet(
				layers=[
						('input', layers.InputLayer),
						('conv1', layers.Conv2DLayer),
						('pool1', layers.MaxPool2DLayer),
						('dropout1', layers.DropoutLayer),
						('conv2', layers.Conv2DLayer),
						('pool2', layers.MaxPool2DLayer),
						('dropout2', layers.DropoutLayer),
						('conv3', layers.Conv2DLayer),
						('pool3', layers.MaxPool2DLayer),
						('dropout3', layers.DropoutLayer),
						('hidden4', layers.DenseLayer),
						('dropout4', layers.DropoutLayer),
						('hidden5', layers.DenseLayer),
						('dropout5', layers.DropoutLayer),
						('output', layers.DenseLayer),
						],


				input_shape=(None, Settings.NN_CHANNELS, Settings.NN_INPUT_SHAPE[0], Settings.NN_INPUT_SHAPE[1]), # variable batch size, 3 color shape row shape

				conv1_num_filters=32, conv1_filter_size=(5, 5), conv1_stride=(1, 1), conv1_pad=(1, 1), conv1_nonlinearity=lasagne.nonlinearities.rectify,
				pool1_pool_size=(2, 2),
				dropout1_p=0.3,

				conv2_num_filters=64, conv2_filter_size=(5, 5), conv2_stride=(1, 1), conv2_pad=(1, 1), conv2_nonlinearity=lasagne.nonlinearities.rectify,
				pool2_pool_size=(2, 2),
				dropout2_p=0.5,

				conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2), conv3_nonlinearity=lasagne.nonlinearities.rectify,
				dropout3_p=0.6,

				hidden4_num_units=512, hidden4_nonlinearity=lasagne.nonlinearities.rectify,
				dropout4_p=0.7,

				hidden5_num_units=512, hidden5_nonlinearity=lasagne.nonlinearities.rectify,
				dropout5_p=0.7,
			
				output_num_units=outputShape, 
				output_nonlinearity=lasagne.nonlinearities.softmax,
			
				# optimization method:
				update=nesterov_momentum,
				update_learning_rate=theano.shared(utils.to_float32(Settings.NN_START_LEARNING_RATE)),
				update_momentum=theano.shared(utils.to_float32(Settings.NN_START_MOMENTUM)),

				batch_iterator_train=AugmentingLazyBatchIterator(Settings.NN_BATCH_SIZE, testData, "train", False, newSegmentation=False, loadingSize=(150,150)),
				batch_iterator_test=LazyBatchIterator(Settings.NN_BATCH_SIZE, testData, "test", False, newSegmentation=False, loadingInputShape=Settings.NN_INPUT_SHAPE),
				train_split=TrainSplit(eval_size=0.0), # we cross validate on our own

				regression=False, # classification problem
				on_epoch_finished=[
					AdjustVariable('update_learning_rate', start=Settings.NN_START_LEARNING_RATE, stop=0.0001),
					AdjustVariable('update_momentum', start=Settings.NN_START_MOMENTUM, stop=0.999),
					TrainingHistory("nouri6_color_do", str(self), [], modelSaver),
					EarlyStopping(350),
					modelSaver,
					],
				max_epochs=Settings.NN_EPOCHS,
				verbose=1,
				)

	def __str__(self):
		return "CNN_I(DA,LR,MO)-C5x5-P2x2-D3-C5x5-P2x2-D5-C2x2-P2x2-D6-D512-D7-D512-D7-O"



