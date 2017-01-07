import random
import numpy as np
import cv2 as cv
from nolearn.lasagne import BatchIterator

import data_io.settings as Settings
from misc import utils


class LazyBatchIterator(BatchIterator):
    """
    This is a lazy implementation of the original BatchIterator.
    The original BatchIterator has the problem that it requires the whole dataset to be loaded into memory.
    However for big datasets and machines with small amounts of RAM this becomes a big problem.
    This Iterator solves this by lazy loading each batch separately. Only use this Iterator on large datasets as this is slower than normal batchLoading.
    """

    def __init__(self, batch_size, testData, segmentIndex, shuffle=False, seed=42, newSegmentation=True, loadingInputShape=Settings.NN_INPUT_SHAPE):
        """ Keyword arguments:
        batch_size -- the size of the batch to load.
        testData -- reference to the test data class
        segmentIndex -- segment index to load from (e.g. train or test)
        shuffle -- shuffle the test data
        seed -- seed for shuffling
        newSegmentation -- do not set to True. This mixes up the train and test segmentations. (FIXME)
        loadingInputShape -- the shape (size) that images should be loaded in
        """

        super(LazyBatchIterator, self).__init__(batch_size, shuffle, seed)
        self.segmentIndex = segmentIndex
        self.testData = testData
        self.newSegmentation = newSegmentation
        self.grayscale = Settings.NN_CHANNELS == 1L

        # unless the subclass AugmentingLazyBatchIterator loadingInputShape will be the size that this class returns
        self.loadingInputShape = loadingInputShape
        # to prepare for loading we have to calculate the mean and std of our testData
        #print "Calculate mean and std of dataset"
        #self.testData.calculate_mean_std(segmentIndex, self.grayscale)
        self.num_samples = 0
        print "# Iterator '{0}' initialized with following settings\n\tNew Segmentation after epoch: {1}\n\tGrayscale: {2}\n\tLoading shape: {3}\n\tOutput shape: {4}\n\tShuffle: {5}\n\tNormalization: {6}".format(segmentIndex, newSegmentation, self.grayscale, self.loadingInputShape, Settings.NN_INPUT_SHAPE, shuffle, Settings.F_APPLY_NORMALIZATION)

    def __call__(self, X, y=None):
        """
        gets called before every batch iteration to shuffle the data.
        we don't care about X or y because we load them ourselves anyway.
        However in 'predict-mode' X contains the images we want to predict so we have to take that into account.
        In 'predict-mode' y is None. In 'normal-mode' y is not None but has the shape (0,)
        """
        # in predict case
        if y is None:
            self.X = X
        else:
            self.X = None
            if self.shuffle:
                self.testData.reshuffle_segment_data(self.segmentIndex)

            # we will also segment our data new if this batchIterator should do it
            if self.newSegmentation:
                self.testData.new_segmentation(True)
        return self

    def __iter__(self):
        """ Iterates over the batch."""

        # prepare batches
        Xb = []
        yb = []

        # if in 'predict-mode' do not iterate our testdata but iterate self.X instead
        if not self.X is None:
            yield self.X, None # y doesn't seem to matter here
        else:    
            for img, class_ in self.testData.load_data(self.segmentIndex, grayscale=self.grayscale, size=self.loadingInputShape, outputActions=False):
                # apply 'min-max-scaling' as feature standardization since z-normalization does not produces values in the range [0, 1] 
                img = utils.rescale_image_01(img)
                Xb.append(img)
                yb.append(self.testData.classes.index(class_))
                # batch finished? -> Yes -> yield
                if len(yb) == self.batch_size:		
                    Xb, yb = self.__convert_to_valid_input(Xb, yb)		
                    yield self.transform(Xb, yb)

                    # reset batch
                    Xb = []
                    yb = []
            # all images are loaded -> if there is a rest yield it
            if len(yb) > 0:
                Xb, yb = self.__convert_to_valid_input(Xb, yb)
                yield self.transform(Xb, yb)

    def __convert_to_valid_input(self, Xb, yb):
        # convert responses to float 
        Xb = np.float32(Xb)
        yb = np.int32(yb)
        
        # reshape to (n, colorChannels, image_w, image_h)
        Xb = Xb.reshape(Xb.shape[0], Settings.NN_CHANNELS, self.loadingInputShape[0], self.loadingInputShape[1])
        yb = yb.reshape(yb.shape[0], )
        return Xb, yb

    @property
    def n_samples(self):
        X = self.X
        if X is None:
            return self.num_samples
        else:
            return len(X)


class AugmentingBatchIterator(BatchIterator):
    """Overwrites "transform" and augments the incoming data."""

    def transform(self, Xb, yb):
        """ Augments Xb data by applying label preserving changes."""

        Xb, yb = super(AugmentingBatchIterator, self).transform(Xb, yb)

        # prepare new Xb array because the resulting image size might vary depending on random crop setting
        Xb_new = np.empty((Xb.shape[0], Xb.shape[1], Settings.NN_INPUT_SHAPE[0], Settings.NN_INPUT_SHAPE[1]), Xb.dtype)


        for i in xrange(Xb.shape[0]):
            
            cropping = False
            operationKeys = Settings.NN_AUGMENTATION_PARAMS.keys()
            workingImage = Xb[i]
            # augment sample with NN_AUGMENT_CHANCE chance for j-th augmentation iteration (j is the len of ops)
            ops = []
            while random.uniform(0.0, 1.0) < Settings.NN_AUGMENT_CHANCE[len(ops)] and Settings.NN_AUGMENT_DATA:  
                operation = operationKeys[random.randint(0, len(operationKeys)-1)]
                ops.append(operation)
                transformation = Settings.NN_AUGMENTATION_PARAMS[operation]

                # prevent two cropping augmentation methods (free rotation and random cropping)
                if operation == 8 or operation == 6:
                    operationKeys.remove(6)
                    operationKeys.remove(8)

                # no extra parameter
                if transformation[1] is None:
                    workingImage = transformation[0](workingImage)
                # 1 extra parameter
                elif len(transformation) == 2:
                    parameterValue = transformation[1][0](transformation[1][1], transformation[1][2])
                    workingImage = transformation[0](workingImage, parameterValue)
                # 2 extra parameters (rotating)
                elif len(transformation) == 3:
                    parameterValue1 = transformation[1][0](transformation[1][1], transformation[1][2])
                    parameterValue2 = transformation[2][0](transformation[2][1], transformation[2][2])
                    workingImage = transformation[0](workingImage, parameterValue1, parameterValue2)
                # 3 extra parameters (cropping)
                elif len(transformation) == 4:
                    cropping = True
                    parameterValue1 = transformation[1][0](transformation[1][1], transformation[1][2])
                    parameterValue2 = transformation[2][0](transformation[2][1], transformation[2][2])
                    parameterValue3 = transformation[3][0](transformation[3][1], transformation[3][2])                    
                    workingImage = transformation[0](workingImage, parameterValue1, parameterValue2, parameterValue3)

            # make sure that image has the right shape
            if not cropping:
                Xb_new[i] = utils.equalize_image_size(Xb[i], Settings.NN_INPUT_SHAPE[0] * Settings.NN_INPUT_SHAPE[1])
            else:
                Xb_new[i] = workingImage

            # To test the actual output
            #print "Applied {0} transformations. Ops: {1}".format(len(ops), ops)
            #saveImgs = utils.radio_question("[?]", "Save images", None, ["Yes", "No"], [True, False])
            #if saveImgs:
            #    afterTrans = np.copy(Xb_new[i])
            #    #print "after trans before:",Xb_new[i].shape
            #    afterTrans, _ = utils.reshape_to_cv_format(afterTrans, True)
            #    #beforeTrans, _ = utils.reshape_to_cv_format(beforeTrans, True)
            #    #print "after trans after:",afterTrans.shape
            #    cv.imwrite(utils.get_temp_path() + "op_after.jpg", afterTrans)
            #    #cv.imwrite(utils.get_temp_path() + "op_before.jpg", beforeTrans)
        xb = None
        return Xb_new, yb


class AugmentingLazyBatchIterator(LazyBatchIterator, AugmentingBatchIterator):
    """ Combination of LazyBatchIterator and AugmentingBatchIterator."""

    def set_iterator_specific_settings(self):
        # calculate translation settings if enabled
        if 6 in Settings.NN_AUGMENTATION_PARAMS:
            # prevent "normal" translation (we do random cropping of a bigger picture)
            del Settings.NN_AUGMENTATION_PARAMS[0]
            del Settings.NN_AUGMENTATION_PARAMS[1]

            horizontalTranslationMargin = self.loadingInputShape[0] - Settings.NN_INPUT_SHAPE[0]
            verticalTranslationMargin = self.loadingInputShape[1] - Settings.NN_INPUT_SHAPE[1]
            Settings.NN_AUGMENTATION_PARAMS[6] = (Settings.NN_AUGMENTATION_PARAMS[6][0], (Settings.NN_AUGMENTATION_PARAMS[6][1][0], Settings.NN_AUGMENTATION_PARAMS[6][1][1], horizontalTranslationMargin), (Settings.NN_AUGMENTATION_PARAMS[6][2][0], Settings.NN_AUGMENTATION_PARAMS[6][2][1], verticalTranslationMargin), (Settings.NN_AUGMENTATION_PARAMS[6][3][0], Settings.NN_INPUT_SHAPE, Settings.NN_INPUT_SHAPE))
        else:
            del Settings.NN_AUGMENTATION_PARAMS[6]
            

    def __init__(self, batch_size, testData, segmentIndex, shuffle = False, seed = 42, newSegmentation = True, loadingSize=Settings.NN_INPUT_SHAPE):  
        super(AugmentingLazyBatchIterator, self).__init__(batch_size, testData, segmentIndex, shuffle, seed, newSegmentation, loadingInputShape=loadingSize)
        self.set_iterator_specific_settings()
        # continue settings output
        print "\tAugmentation Params:"
        for op in Settings.NN_AUGMENTATION_PARAMS:
            print "\t\t",Settings.NN_AUGMENTATION_PARAMS[op]
               











