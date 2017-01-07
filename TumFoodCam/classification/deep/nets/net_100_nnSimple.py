import theano
import lasagne
from lasagne import layers, nonlinearities
from nolearn.lasagne import NeuralNet, TrainSplit
from lasagne.updates import nesterov_momentum
from classification.deep.learning_functions import AdjustVariable, EarlyStopping
from classification.deep.training_history import TrainingHistory
from classification.deep.batch_iterators import LazyBatchIterator, AugmentingLazyBatchIterator

from misc import utils
import data_io.settings as Settings
import numpy as np


class net_100_nnSimple(object):
	"""
	Simple Neural Net
	Input       32x32
	Hidden1     500
	Hidden1     50
	Output      X
	"""
	
	def set_network_specific_settings(self):
		Settings.NN_INPUT_SHAPE = (32, 32)
		Settings.NN_CHANNELS = 1L
		Settings.NN_EPOCHS = 3000
		# grayscale so let's not use brightness or saturation changes (last two entries)
		if 4 in Settings.NN_AUGMENTATION_PARAMS:
			del Settings.NN_AUGMENTATION_PARAMS[4]
		if 5 in Settings.NN_AUGMENTATION_PARAMS:
			del Settings.NN_AUGMENTATION_PARAMS[5]


	def __init__(self, outputShape, testData, modelSaver):
		self.set_network_specific_settings()
		modelSaver.model = self
		self.net = NeuralNet(
			layers=[
				('input', layers.InputLayer),

				('hidden1', layers.DenseLayer),
				('hidden2', layers.DenseLayer),

				('output', layers.DenseLayer),
				],

			# Layer parameter
			input_shape=(None, Settings.NN_CHANNELS, Settings.NN_INPUT_SHAPE[0], Settings.NN_INPUT_SHAPE[1]), # variable batch size, single row shape
			hidden1_num_units=500,
			hidden2_num_units=50,
			output_num_units=outputShape, 
			output_nonlinearity=lasagne.nonlinearities.softmax,
				
			# optimization method:
			update=nesterov_momentum,
			update_learning_rate=theano.shared(utils.to_float32(Settings.NN_START_LEARNING_RATE)),
			update_momentum=theano.shared(utils.to_float32(Settings.NN_START_MOMENTUM)),

			batch_iterator_train=AugmentingLazyBatchIterator(Settings.NN_BATCH_SIZE, testData, "train", True, loadingSize=(50,50)),
			batch_iterator_test=LazyBatchIterator(Settings.NN_BATCH_SIZE, testData, "valid", False, newSegmentation=False),
			train_split=TrainSplit(eval_size=0.0), # we cross validate on our own


			regression=False, # classification problem
		on_epoch_finished=[
				AdjustVariable('update_learning_rate', start=Settings.NN_START_LEARNING_RATE, stop=0.0001),
				AdjustVariable('update_momentum', start=Settings.NN_START_MOMENTUM, stop=0.999),
				TrainingHistory("?", str(self), [], modelSaver),
				EarlyStopping(150),
				modelSaver,
				],
			max_epochs=Settings.NN_EPOCHS,
			verbose=1,
			)

	def fit(self, x, y):
		return self.net.fit(x, y)

	def __str__(self):
		return "NN_I-D500-D50-O"


