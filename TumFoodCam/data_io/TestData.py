from os import walk, remove
import os.path as osPath
import numpy as np
import itertools
import cv2 as cv
from random import randint, sample
import random
from misc import utils
import data_io.settings as Settings
import data_io.mail_service as MailServer
import sys
from math import sqrt
import time
from segmentation.image_region import ImageRegion
import logging

def load_image(fileName, grayscaleFlag, resizeFactor):
    img = cv.imread(fileName, grayscaleFlag)
    # img is None if the image could not be read. imread cannot read .gif for example
    if img is None:
        return img

    if resizeFactor != 1:                    
        return utils.resize_image(img, resizeFactor)
    else:
        return img

def sliding_window(image, size, stride):
    for y in xrange(0, image.shape[0], stride):
        for x in xrange(0, image.shape[1], stride):
            roi = ImageRegion(upperLeft=(x,y), lowerRight=(x+size[0],y+size[1]))
            yield roi, utils.crop_image(image, x, y, size[0], size[1])


class TestData(object):
    """description of class"""
    
    # creates a 8 fold cross validation.
    # this means that the test data gets segmented into 8 parts and each run changes one part.
    # 0 means no cross validation
    # Since we use k-fold cross validation this variable would be k.
    DEFAULT_CROSS_VALIDATION_LEVEL = 8

    def __init__(self, rootPath, crossValidation=-1, shuffle=False, seed=42):
        self.shuffle = shuffle
        self.seed = seed
        self.set_root_path(rootPath)
        self.reset_data_set()

        self.mean = {} # keys are the image sizes
        self.meanScaled = {}
        self.std = {}
        self.stdScaled = {}
        # crossValidation = -1 means default cross validation level
        if crossValidation == -1:
            self.crossValidationLevel = TestData.DEFAULT_CROSS_VALIDATION_LEVEL
        else:
            self.crossValidationLevel = crossValidation
        
        if Settings.G_DETAILED_CONSOLE_OUTPUT:
            print len(self.__classDictionary), "classes initialized. Total Size:", self.testDataSize


    def reset_data_set(self):
        self.segmentRatio = {}
        self.segmentSizeMean = {}
        self.numberOfSlices = 0

        # Contains the "raw" segments for each class.
        # new_segmentation takes out the segments to form the segments used for training and validation
        # Format: {class_: [ [ ] ] }
        #            |     |  |
        #            |     | file list
        #            |     list of segments
        #           Dict of classes
        self.__slicedClassDictionary  = {}

        # dict that stores the current indices for segmentation
        self.__segmentSliceIndex = {}
        self.crossValidationIteration = 0

    def set_root_path(self, rootPath):
        """Sets the root path for the test data.
        """

        # Append '/' at the end if it does not exist.
        if not rootPath.endswith("/"):
            rootPath += "/"
        self.__rootPath = rootPath

        # Initialize samples with new root Path.
        self.__classDictionary = {}
        self.testDataSize = 0
        self.numberOfClasses = 0
        # Generate class dictionary.
        self.__generate_class_dictionary()
        return self.testDataSize
    
    def get_root_path(self):
        return self.__rootPath
        
    def __generate_class_dictionary(self):
        # Create list of folders in the root path
        folderList = []
        try:
            folderList = next(walk(self.get_root_path()))[1]
        except:
            logging.exception("Could not iterate through root folder of dataset. - Path: {0}.".format(self.get_root_path()))
        
        # in case the test data folder does not contain folders only images
        if len(folderList) == 0:
            logging.warning("Unusual dataset. - Path: {0}.".format(self.get_root_path()))
            folderList = [""]   
            
        self.numberOfClasses = len(folderList)
        self.classes = folderList

        

        print self.numberOfClasses," classes found. Loading file names."

        # Create the dictionary. Key is the folder Name e.g. pizza.
        # value will be a list of image names (file names) for this class.
        self.__classDictionary = {className: [] for className in folderList}

        

        self.numberOfImagesPerClass = {}
        for className in self.__classDictionary:
            try:
                (_, _, self.__classDictionary[className]) = walk(self.get_root_path() + className).next()
            except:
                logging.exception("Could not iterate through folder {0}.".format(self.get_root_path() + className))

            # filter by extension. Only allow jpg, JPG, Jpg, png, PNG, Png
            self.__classDictionary[className] = [ file for file in self.__classDictionary[className] if file.endswith( ('.jpg','.Jpg','.JPG','.png','.Png','.PNG') ) ]

            # shuffel if shuffle flag is set
            if self.shuffle:
                random.seed(self.seed)
                random.shuffle(self.__classDictionary[className])
            self.numberOfImagesPerClass[className] = len(self.__classDictionary[className])
            self.testDataSize += self.numberOfImagesPerClass[className]

    def segment_test_data(self, segments={"train": 7, "test": 1}):
        """Segment each class according to the values in segment.

        The default value for segments is the default for k-fold segmentation which takes k-1 parts as the training set
        and 1 part as the testing set.

        Keyword argument 
        segments -- the segmentation of the available data. The sum of the segment values has to be <= crossValidationLevel.
        """
        
        # if cross validation is enabled the segment values can not be bigger than the validation level
        if self.crossValidationLevel > 1 and sum(segments.itervalues()) > self.crossValidationLevel:
            raise AttributeError("Sum of segment sizes > crossValidationLevel")

        # copy segments dict as it is a direct reference to the settings TESTDATA_SEGMENTS dict.
        # changing segments changes this value as well
        segments = segments.copy()

        # in how many parts / slices should the samples be divided into
        numberOfSlices = self.crossValidationLevel
        if self.crossValidationLevel <= 1:
            # no cross validation -> numberOfSlices: 10 which enables values like 0.3 or 0.7 in segments (would be 3 parts and 7 parts)
            numberOfSlices = 10
            # to keep "The sum of the segment values has to be <= crossValidationLevel." truth the segments have to be e.g. 0.3 & 0.7 so we have to 
            # convert them to 3 and 7
            for segmentName, value in segments.iteritems():
                segments[segmentName] = int(value * 10)
            
        self.segmentRatio = segments
        self.numberOfSlices = numberOfSlices

        # Holds the mean number of samples for each segment. (initialization)
        self.segmentSizeMean = dict((key, 0) for key in segments)
        for class_, samples in self.__classDictionary.iteritems():    
            sampleIndex = 0

            # slice the data in samples into k (crossValidationLevel) parts with sampleLength / k size
            slicedSamples = []
            sliceLength = float(len(samples)) / float(numberOfSlices)

            #calc the size of the segments
            for segment in self.segmentSizeMean:
                self.segmentSizeMean[segment] += round(segments[segment] * sliceLength)

            for i in xrange(numberOfSlices):
                # we round sliceLength because otherwise we would loose images. (If sliceLength is 9.9 and we take 9 we would miss images at the end)
                # if we ceil (have 9.5 and take 10) the last segment has to be smaller
                # if we floor (have 9.4 and take 9) the last segmetn has to be bigger
                # because of this we have to treat the last slice differently
                if i == numberOfSlices - 1:
                    segment = samples[sampleIndex:]
                else:
                    segment = samples[sampleIndex:sampleIndex+int(round(sliceLength))]
                
                sampleIndex += int(round(sliceLength))
                slicedSamples.append(segment)

            # add to __slicedClassDictionary
            self.__slicedClassDictionary[class_] = slicedSamples
            
        # Print result and calculate mean images per segment
        for segment in self.segmentSizeMean:
            self.segmentSizeMean[segment] /= self.numberOfClasses
        #if Settings.G_DETAILED_CONSOLE_OUTPUT:
        #    print "Segmentation complete:"
        #    print "Segments:",self.segmentSizeMean        

    def new_segmentation(self, infiniteSegmentation=False):
        if self.crossValidationIteration >= self.crossValidationLevel and not infiniteSegmentation:
            # finished :)
            return False

        self.crossValidationIteration += 1

        # first segmentation?
        if not self.__segmentSliceIndex:
            # just take the size of the segment and add it to the index for the starting segmenation
            startIndex = 0
            for segmentName, size in self.segmentRatio.iteritems():
                endIndex = startIndex + size - 1
                self.__segmentSliceIndex[segmentName] = (startIndex, endIndex)
                startIndex += size
        # already segmented
        else:
            for segmentName, (startIndex, endIndex) in self.__segmentSliceIndex.iteritems():
                segmentSize = self.segmentRatio[segmentName]
                # would the new index overflow the bounds?
                if startIndex + 1 == self.numberOfSlices:
                    startIndex = 0
                    endIndex = segmentSize - 1
                else:
                    startIndex += 1

                    # would the new end index overflow the bounds?
                    if endIndex + 1 == self.numberOfSlices:
                        endIndex = 0
                    else:
                        endIndex += 1

                self.__segmentSliceIndex[segmentName] = (startIndex, endIndex)


        # based on self.__segmentSliceIndex divide the samples to the segments like train, test etc.
        for class_, samples in self.__classDictionary.iteritems():            
            # holds the segments (key) and the images (value) for class_
            segmentedClassDictionary = {}

            for segmentName, (startIndex, endIndex) in self.__segmentSliceIndex.iteritems():
                segments = []
                # go on till we reach the endIndex (could be lower than the startIndex)
                index = startIndex
                while(True):
                    try:
                        segment = self.__slicedClassDictionary[class_][index]
                    except:
                        logging.exception("Warning! Could not segment ClassDictionary")
                        return False
                    segments.extend(segment)

                    # did we reach the endIndex?
                    if index == endIndex:
                        break

                    # overflow bounds with new index?
                    if index + 1 == self.numberOfSlices:
                        index = 0
                    else:
                        index += 1                    

                # segments holds all samples for this segment in this class
                segmentedClassDictionary[segmentName] = segments

            # segmentedClassDictionary holds all the segments for this class
            self.__classDictionary[class_] = segmentedClassDictionary

        return True

    def reshuffle_segment_data(self, segmentIndex):
        for class_ in self.__classDictionary:            
            random.shuffle(self.__classDictionary[class_][segmentIndex]) 
                
    def load_data(self, segmentIndex, numberOfClasses=None, classes=[], grayscale=True, resizeFactor=1, outputActions=True, maxNumberOfImagesPerClass=-1, yieldFilename=False, size=(-1,-1), transformation=None, resolutionSize=-1, forceNormalization=0, forceZNormalization=0, forceZcaWhitening=0):
        """Loads the images of the test data as a generator.

        Keyword argument 
        segmentIndex -- the segment index to load from. If default values were used in segment_test_data use "train" or "test"
        numberOfClasses -- the number of classes to load (default None (load all classes))
        classes -- explicitly specified classes (default [])
        grayscale -- should images be loaded in grayscale or not (default True)
        maxNumberOfImagesPerClass -- limits the loading of images (default -1 means no limit)
        """
        #Parameter validation
        if segmentIndex not in self.segmentRatio and segmentIndex != "all":
            raise AttributeError(segmentIndex + " is not in segments")

        # test if size is valid
        try:
            sizeLen = len(size)
            if sizeLen != 2:
                raise AttributeError("size has to be a 2-element tuple. Use (-1, -1) if you don't care.")
        except:
            raise AttributeError("size has to be a 2-element tuple. Use (-1, -1) if you don't care.")             

        desired_h = size[0]
        desired_w = size[1]  

        if (desired_h / desired_w) != 1.0:
            raise AttributeError("The current version does not support an aspect ratio other than 1.0.")
        
        if resizeFactor != 1 and resolutionSize != -1:
            raise AttributeError("You can't set resizeFactor and resolutionSize at the same time.")

        # if forceX parameter is 0 it means do not change. -1 -> force false, +1 -> force true
        forceNormalization = (forceNormalization == 0 and Settings.F_APPLY_NORMALIZATION) or forceNormalization == 1
        forceZNormalization = (forceZNormalization == 0 and Settings.F_APPLY_ZNORMALIZATION) or forceZNormalization == 1

        if (forceNormalization and forceZNormalization):
            raise AttributeError("You can't apply normalization (mean substraction) and z-normalization (normalization divided by std) at the same time.")


        # calculate mean and std if normalization enabled and needed
        if forceNormalization or forceZNormalization:
            if desired_h == -1:
                if resolutionSize == -1:
                    raise AttributeError("If F_APPLY_NORMALIZATION or F_APPLY_ZNORMALIZATION is enabled the image must have a fixed size or at least a fixed resolution size.")
                else:
                    desired_h = desired_w = int(sqrt(resolutionSize))
                    size = (desired_w, desired_h)
                

            if len(self.mean) == 0 or len(self.std) == 0:
                self.__calculate_mean_std(size, grayscale)

        preprocessImage = forceNormalization or forceZNormalization or Settings.F_APPLY_ZCA_WHITENING or Settings.F_APPLY_CLAHE or Settings.F_APPLY_HISTOGRAM_EQ

        if numberOfClasses is None:
            numberOfClasses = self.numberOfClasses
        else:
            numberOfClasses = min(numberOfClasses, self.numberOfClasses)

        if not classes:
            #Fill classes with numberOfClasses count classes
            classes = [key for key in self.__classDictionary]
            # only take numberOfClasses classes
            classes = classes[:numberOfClasses]

        limitImageLoading = True
        if maxNumberOfImagesPerClass == -1:
            limitImageLoading = False
        else: 
            maxNumberOfImagesPerClass = max(1, maxNumberOfImagesPerClass) # load at least one image


        #Load flag for cv.imread.
        loadFlag = cv.IMREAD_GRAYSCALE if grayscale else cv.IMREAD_UNCHANGED
        if outputActions:
            print "Loading dataset {0} {1}. |Classes: {2}| - number of images per segment: {3}".format(segmentIndex, self.__segmentSliceIndex[segmentIndex], numberOfClasses, self.segmentSizeMean[segmentIndex])
            print "Loading in grayscale",grayscale
        for class_ in classes:
            segmentedSamples = self.__classDictionary[class_]
            
            if segmentIndex != "all":
                samples = segmentedSamples[segmentIndex]
            else:
                samples = []
                for segId in self.__classDictionary[class_]:
                    samples.extend(self.__classDictionary[class_][segId])

            if limitImageLoading:
                samplesToLoad = min(maxNumberOfImagesPerClass, len(samples))
            else:
                samplesToLoad = len(samples)            
            if outputActions:
                print "\n"
            
            errors = 0
            lastError = ""
            for i in xrange(samplesToLoad):
                filename = self.get_root_path() + class_ + "/" + samples[i]
                img = load_image(filename, loadFlag, resizeFactor)
                # img is None if the image could not be read. imread cannot read .gif for example
                if img is None:
                    errors += 1
                    lastError = "Image {0} was None.\nSamples:{1}".format(filename, samples)
                    continue

                # if image has an alpha channel reduce
                if len(img.shape) > 2 and img.shape[2] == 4L:
                    img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

                # do we need to crop and adjust the size
                if desired_h != -1 and desired_w != -1:
                    img = utils.crop_to_square(img)
                    desiredArea = desired_h * desired_w
                    img = utils.equalize_image_size(img, desiredArea)
                    if img is None:
                        errors += 1
                        lastError = "Image {0} after eq-1 was None".format(filename)
                        continue
                    if desired_w != img.shape[1] or desired_h != img.shape[0]:
                        img = utils.crop_around_center(img, desired_w, desired_h)
                        if img is None:
                            errors += 1
                            lastError = "Image {0} after cropping was None".format(filename)
                            continue

                # resize image to set size
                if resolutionSize != -1 and size[0] == -1 and size[1] == -1:
                    try:
                        img = utils.equalize_image_size(img, resolutionSize)
                    except:
                        logging.exception("Could not eq image size: img.shape: {0} - new size: {1}".format(img.shape, resolutionSize))
                        lastError = "Image eq-2 exception".format(filename)
                        img = None
                    if img is None:
                        lastError = "Image {0} after eq-2 was None".format(filename)
                        errors += 1
                        continue


                if not transformation is None:
                    try:                        
                        img = transformation(img)
                    except Exception, e:
                        errors += 1
                        lastError = "Exception during image transformation:",e.message
                        continue

                if preprocessImage:
                    try:                        
                        img = self.__preprocess_image(img, forceNormalization, forceZNormalization, grayscale)
                    except Exception, e:
                        errors += 1
                        lastError = "Exception during image preprocessing:",e.message
                        continue


                utils.show_progress(outputActions, i+1, samplesToLoad, "Loading {0} images in class {1} (Loss: {2}):", samplesToLoad, class_, errors)
                if yieldFilename:
                    yield (img, class_, samples[i])
                else:
                    yield (img, class_)

            if errors == samplesToLoad:
                print "Could not load any samples for class {0}. Segment: {1}\n\t{2} Errors - {3} Samples\n\tLast Error: {4}".format(class_, segmentIndex, errors, samplesToLoad, lastError)

    def load_segment(self, segmentIndex, grayscale=True, size=(-1, -1), transformation=None):
        images = []
        classes = []
        for img, class_ in self.load_data(segmentIndex, grayscale=grayscale, size=size, outputActions=False, transformation=transformation, forceNormalization=-1, forceZNormalization=-1):
            images.append(img)
            classes.append(self.classes.index(class_))     
        
        images_np = np.array(images)
        return (images_np, classes)

    def load_segment_dummy(self, segmentIndex, size):
        # get the exact number of images for segment
        # unfortunatelly we do have to dummy-load them because some images might not load later on,
        # and we need to make sure that we have the exact number of loadable images.
        datasetSize = 0
        for _, _ in self.load_data(segmentIndex, grayscale=True, size=(2,2), outputActions=False, forceNormalization=-1, forceZNormalization=-1):
            datasetSize += 1
        return np.empty((datasetSize, Settings.NN_CHANNELS, size[0], size[1])), np.empty((datasetSize,))

    def __preprocess_image(self, image, normalization, zNormalization, grayscale):
        if Settings.F_APPLY_HISTOGRAM_EQ and Settings.F_APPLY_CLAHE:
            logging.warning("It is not advised to apply CLAHE and Histogram EQ at the same time.")

        if Settings.F_APPLY_HISTOGRAM_EQ:
            if grayscale:
                image = utils.equalize_image_channel(image)
            else:
                image = utils.equalize_BGR_image(image)

        if Settings.F_APPLY_CLAHE:
            if grayscale:
                image = utils.equalize_image_channel_adaptive(image)
            else:
                image = utils.equalize_BGR_image_adaptive(image)

        if normalization:
            image = self.apply_normalization(image)

        if zNormalization:
            image = self.apply_z_normalization(image)

        if Settings.F_APPLY_ZCA_WHITENING:
            image = utils.ZCA_whitening(image)

        return image

    def __calculate_mean_std(self, size, grayscale):
        print "** Calculating mean and std image for dataset segment {0} with size {1} **".format("all", size)
        images, _ = self.load_segment("all", grayscale, size, transformation=None)
        
        imageIndex = size[0] * size[1]

        self.mean[imageIndex] = np.mean(images, axis=0)        
        self.std[imageIndex] = np.std(images, axis=0)

        # scale to [0, 1]
        self.meanScaled[imageIndex] = self.mean[imageIndex] / 255
        self.stdScaled[imageIndex] = self.std[imageIndex] / 255
        print "** Calculation finished **"
    
    def apply_z_normalization(self, image, dtype=np.uint8):
        imageIndex = image.shape[0] * image.shape[1]
        
        # image is scaled to [0, 1] apply scaled values
        if np.max(image) <= 1.0:
            return self.apply_normalization(image, -1) / self.stdScaled[imageIndex]
        
        image = self.apply_normalization(image, -1) / self.std[imageIndex]

        # don't change dtype if -1
        if dtype == -1:
            return image
        return image.astype(dtype)

    def apply_normalization(self, image, dtype=np.uint8):
        """ Substracts the mean.
        """
        imageIndex = image.shape[0] * image.shape[1]
        if len(self.mean) == 0 or len(self.std) == 0:
            raise Exception("Mean and Std image are not computed")

        # test if mean or std have been calculated for this image size
        if not imageIndex in self.mean:
            #raise Exception("The shape of the mean image and the input image do not match. mean.shape: {0} - image.shape: {1}".format(self.mean.shape, image.shape))
            # resize to fit image.
            meanImage = (self.mean.values())[0]
            stdImage = (self.std.values())[0]
            self.mean[imageIndex] = utils.equalize_image_size(meanImage, imageIndex)
            self.std[imageIndex] = utils.equalize_image_size(stdImage, imageIndex)

        # make sure image type is not uint8
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # image is scaled to [0, 1] apply scaled values
        if np.max(image) <= 1.0:
            return image - self.meanScaled[imageIndex]

        image -= self.mean[imageIndex]
        # don't change dtype if -1
        if dtype == -1:
            return image
        image = image.astype(dtype)
        #cv.imshow("Mean", image)
        #cv.waitKey(0)
        return image
    
    def choose_image(self, segmentIndex, grayscale=True, classes=[], resizeFactor=1):
        """ Gives the user the possibility to choose from a set of images.
        """ 
        
        if segmentIndex not in self.segmentRatio:
            raise AttributeError(segmentIndex + " is not in segments")

        if not classes:
            classes = self.classes

        #Load flag for cv.imread.
        loadFlag = cv.IMREAD_GRAYSCALE if grayscale else cv.IMREAD_UNCHANGED

        # select class
        class_ = classes[randint(0, len(classes) - 1)]

        images = []
        number = 1

        images = [img for img, _ in self.load_data(segmentIndex, 1, [class_], grayscale, resizeFactor, False)]
        titles = range(1, len(images) + 1)
        print "Choose the image to select, close the dialog and enter the number."

        imgNumber = 0
        while(True):
            utils.display_images(images, titles=titles, columns=8)
            
            input = raw_input("Pick image number or type 'again' to see them again: ")
            if input == "again":
                continue
            else:
                try:

                    imgNumber = int(input)                    
                except:
                    print "Please insert a valid number."
                    continue

                if imgNumber <= 0:
                        print "Please insert a valid number bigger than zero."
                        continue
                else:
                    break
        return images[imgNumber - 1]

    def pick_random_image(self, segmentIndex, grayscale=True, resizeFactor=1):
        """ Selects a random image from the selected segment.
        """

        if segmentIndex not in self.segmentRatio:
            raise AttributeError(segmentIndex + " is not in segments")

        #Load flag for cv.imread.
        loadFlag = cv.IMREAD_GRAYSCALE if grayscale else cv.IMREAD_UNCHANGED

        # select class
        class_ = self.classes[randint(0, self.numberOfClasses - 1)]

        # select sample
        segmentedSamples = self.__classDictionary[class_]
        samples = segmentedSamples[segmentIndex]
        samplePath = self.get_root_path() + class_ + "/" + samples[randint(0, len(samples) - 1)]
        return (class_, load_image(samplePath, loadFlag, resizeFactor))

    def export_test_data_information(self):
        output = []
        outputTitle = ["segment", "path", "number of classes", "number of images", "segment ratio", "random chance impr."]
        outputTitle.extend(["number of images in " + class_ for class_ in self.classes])
        output.append(outputTitle)

        for segment in self.segmentRatio:
            segmentRow = [segment, "-", self.numberOfClasses, self.segmentSizeMean[segment]*self.numberOfClasses, self.segmentRatio[segment], "-"]
            for class_ in self.classes:
                segmentRow.append(len(self.__classDictionary[class_][segment]))
            output.append(segmentRow)

        # append "all"
        total = ["all", self.get_root_path(), self.numberOfClasses, self.testDataSize, self.crossValidationLevel, self.null_score]
        total.extend(self.numberOfImagesPerClass.itervalues())
        output.append(total)
        return output

    @property
    def mean_image(self):
        if len(self.mean) == 0:
            return None
        return (self.mean.itervalues())[0].astype(np.uint8)

    @property
    def std_image(self):
        if len(self.std) == 0:
            return None
        return (self.std.itervalues())[0].astype(np.uint8)
    
    @property
    def random_score(self):
        """ assuming a random distribution, what would be the score.
            This is only accurate if the total number of images is distrubuted evenly across all classes
        """
        # stores classes as key and the prediction results (list) as value.
        return float(1) / float(self.numberOfClasses)

    @property
    def null_score(self):
        """ assuming a random distribution, what would be the score.
            Since this method takes the largest class as the basis for the calclulation it works for uneven testdata sets.
        """
        return float(max(self.numberOfImagesPerClass.values())) / float(self.testDataSize)

    def add_prefix_to_test_data(self, prefix):
        # terrible code. I know renaming is much easier but this was simple copy paste from normalize_test_data


        # segment data with only one segment
        self.segment_test_data({"all": 1})
        self.new_segmentation()

        numberOfImages = self.testDataSize
        numberOfImagesDone = 0        

        print "Starting renaming to {0}_XX.jpg.\n".format(prefix)
        for img, class_, fileName in self.load_data("all", grayscale=False, outputActions=False, yieldFilename=True):
            oldPath = self.get_root_path() + "/" + class_ + "/" + fileName    
            path = self.get_root_path() + "/" + class_ + "/" + prefix + "_" + fileName    
            remove(oldPath)
            cv.imwrite(path, img)

            numberOfImagesDone += 1
            utils.show_progress(True, numberOfImagesDone, numberOfImages, "Processing class {0}.\tTotal progress:", class_)
        print "Renaming finished."

    def crop_bounding_boxes(self, boundingBoxFileName):        
        # segment data with only one segment
        self.segment_test_data({"all": 1})
        self.new_segmentation()

        numberOfImages = self.segmentSizeMean["all"] * self.numberOfClasses
        numberOfImagesDone = 0        
        currentClass = ""

        print "Starting bounding box cropping \n"
        for img, class_, fileName in self.load_data("all", grayscale=False, outputActions=False, yieldFilename=True):
            path = self.get_root_path() + "/" + class_ + "/"
            if not currentClass == class_:
                currentClass = class_

                # contains all bounding boxes for a class. Key is the image id
                boundingBoxes = {}

                # find bounding box file
                try:
                    with open(path + boundingBoxFileName, "r") as f:
                        firstLine = True
                        for line in f:
                            # skip the first line because the first line only contains the header
                            if firstLine:
                                firstLine = False
                                continue
                            data = line.split()
                            boundingBoxes[data[0]] = ImageRegion(upperLeft=(int(data[1]), int(data[2])), lowerRight=(int(data[3]), int(data[4])))
                except:
                    logging.exception("Could not open bounding box file under path {0}.".format(str(path + boundingBoxFileName)))


            # crop bounding box
            fileId = osPath.splitext(fileName)[0]
            if not fileId in boundingBoxes:
                print "Could not find bounding box for image",fileName
                continue
            else:
                bb = boundingBoxes[fileId]
                try:
                    img = bb.crop_image_region(img)
                except:
                    logging.exception("Could not crop bounding box for image id {0}. Bounding Box: {1} - {2}".format(fileId, bb.upperLeft, bb.lowerRight))
              
            path += fileName    
            cv.imwrite(path, img)

            numberOfImagesDone += 1
            utils.show_progress(True, numberOfImagesDone, numberOfImages, "Processing class {0}.\tTotal progress:", class_)
        print "Cropping finished."
        print "\n"

        if Settings.G_MAIL_REPORTS:
            MailServer.send_mail("", "cropping finished")

        raw_input("Press any key to continue.")
    
    def normalize_test_data(self, size, newName="", forceOverwrite=False):
        normTestDataRootPath = utils.get_parent_dir(self.get_root_path()) + "/"
        if newName == "":
            if not forceOverwrite:
                overwrite = utils.radio_question("[?]", "Do you really wish to overwrite existing images?", None, ["Yes", "No"], [True, False])
            else:
                overwrite = True
            if not overwrite:
                normTestDataRootPath += utils.value_question("", "Provide new foldername:", "s")
            else:
                normTestDataRootPath = self.get_root_path()
        else:
            normTestDataRootPath += newName

        utils.create_dir_if_necessary(normTestDataRootPath)
        print "Saving equalized test data set in path",normTestDataRootPath

        # segment data with only one segment
        self.segment_test_data({"all": 1})
        self.new_segmentation()

        numberOfImages = self.testDataSize
        numberOfImagesDone = 0
        
        currentClass = ""

        print "Starting equalization.\n"
        for img, class_, fileName in self.load_data("all", grayscale=False, outputActions=False, yieldFilename=True):

            path = normTestDataRootPath + "/" + class_ + "/" 
            # reset counter if new class
            if not currentClass == class_:
                currentClass = class_                
                utils.create_dir_if_necessary(path)
                                
            resizedImg = utils.equalize_image_size(img, size)
            path += fileName
            cv.imwrite(path, resizedImg)
            numberOfImagesDone += 1
            utils.show_progress(True, numberOfImagesDone, numberOfImages, "Processing class {0}.\tTotal progress:", currentClass)
        print "\nEqualization finished."
        print "\n"

    def crop_test_data_to_square(self, manuallyDecideFolderName):
        self.segment_test_data({"all": 1})
        self.new_segmentation()

        numberOfImages = self.testDataSize
        numberOfImagesDone = 0        
        rejectedImages = 0

        manDir = utils.get_parent_dir(self.get_root_path()) + "/" + manuallyDecideFolderName

        print "Starting cropping to square aspect ratio. Files that can't be processed automatically will be saved in path {0}.\n".format(manDir)

        currentClass = ""
        for img, class_, fileName in self.load_data("all", grayscale=False, outputActions=False, yieldFilename=True):
            currentFilePath = self.get_root_path() + "/" + class_ + "/" + fileName   

            if not currentClass == class_:
                currentClass = class_
                manDir = utils.get_parent_dir(self.get_root_path()) + "/" + manuallyDecideFolderName + "/" + class_ + "/" 
                utils.create_dir_if_necessary(manDir)

            croppedImg = utils.crop_to_square(img)
            if croppedImg is None:
                # could not crop image to square because aspect ration was to big / small
                # save to path were we have to decide manually and remove the other image
                cv.imwrite(manDir + fileName, img)
                remove(currentFilePath)
                rejectedImages += 1
            else:
                cv.imwrite(currentFilePath, croppedImg)
            numberOfImagesDone += 1
            utils.show_progress(True, numberOfImagesDone, numberOfImages, "Processing class \t{0}.\tRejected images:{1}\tTotal progress:", class_, rejectedImages)
        print "\n\nCropping finished. Rejected images:{0}".format(rejectedImages)
        print "\n"

        if Settings.G_MAIL_REPORTS:
            MailServer.send_mail("Rejected images:{0}".format(rejectedImages), "cropping finished")

        raw_input("Press any key to continue.")

    def augment_test_data(self, newName, cherryPickIteration, equalizeAfter2ndIterationSize):

        # functions that get applied during 1st/2nd iteration   
        firstIterationOps = [(utils.flip_image_horizontal, []), (utils.flip_image_vertical, []), (utils.rotate_image, [-90, -180, 90]), (utils.equalize_BGR_image, [])]

        # second iteration        
        possibleAngleValues = range(5, 355)
        secondIterationOps = [(utils.rotate_image, possibleAngleValues)]

        # third iteration
        possibleLightValues = range(-70, -20, 5)
        possibleLightValues.extend(range(25, 75, 5))
        thirdIterationOps = [(utils.change_brightness, possibleLightValues), (utils.change_saturation, possibleLightValues)]

        normTestDataRootPath = utils.get_parent_dir(self.get_root_path()) + "/" + newName
        # when cherrypicking take the same path as original
        if newName == "":
            normTestDataRootPath = self.get_root_path()
        
        
        utils.create_dir_if_necessary(normTestDataRootPath)
        print "Saving new test data set in path",normTestDataRootPath

        firstIterationData = [1, "-", "-", "-", "-"]
        secondIterationData = [2, "-", "-", "-", "-"]
        thirdIterationData = [3, "-", "-", "-", "-"]

        if cherryPickIteration == 1 or cherryPickIteration == -1:
            # First iteration
            startTime = time.clock()
            imagesBefore = self.testDataSize
            numberOfnewImagesDone = self.__augment_test_data_iteration(normTestDataRootPath, 1, firstIterationOps, (2,2), True, True)
            firstIterationData = [1, imagesBefore, numberOfnewImagesDone, self.testDataSize, time.clock() - startTime]

        # Second iteration
        if cherryPickIteration == 2 or cherryPickIteration == -1:
            startTime = time.clock()
            imagesBefore = self.testDataSize
            numberOfnewImagesDone = self.__augment_test_data_iteration(normTestDataRootPath, 2, secondIterationOps, (0,3), False, False)
            secondIterationData = [2, imagesBefore, numberOfnewImagesDone, self.testDataSize, time.clock() - startTime]

        # The thrid iteration contains light changes. For big images those take a really long time.
        # By equalization we can reduce the size before.
        if equalizeAfter2ndIterationSize > 0:
            self.normalize_test_data(equalizeAfter2ndIterationSize, "", True)
            self.reset_data_set()
            self.set_root_path(normTestDataRootPath)

        if cherryPickIteration == 3 or cherryPickIteration == -1:
            # Third iteration
            startTime = time.clock()
            imagesBefore = self.testDataSize
            numberOfnewImagesDone = self.__augment_test_data_iteration(normTestDataRootPath, 3, thirdIterationOps, (0,0), False, False)
            thirdIterationData = [3, imagesBefore, numberOfnewImagesDone, self.testDataSize, time.clock() - startTime]
        results = utils.get_table(["It", "IMGs Before", "New IMGs", "IMGs After", "Elapsed Time"], 1, firstIterationData, secondIterationData, thirdIterationData)
        print "All operations finished.\n\n"
        print results
        print "\n"

        if Settings.G_MAIL_REPORTS:
            MailServer.send_mail(results.get_html_string(), "Increasing finished")
        raw_input("Press any key to continue.")
                
    def __augment_test_data_iteration(self, normTestDataRootPath, iteration, iterationOps, numberOfOps, uniqueOps=True, saveOriginalImage=False):
        """
        normTestDataRootPath: root path for the dataset
        iteration: number of current iteration (only cosmetic)
        iterationOps: list of iteration operation tuples [(function, [possibleParams])]
        numberOfOps: number of ops to perform - format tuple range (min, max)
        uniqueOps: should operations be unique (only one op of this type per image)
        saveOriginalImage: should we save the original image
        """
        print "\n\nIteration {0}:\n".format(iteration)

         # segment test data
        self.segment_test_data({"all": 1})
        self.new_segmentation()

        numberOfImages = self.testDataSize
        numberOfImagesDone = 0
        numberOfnewImagesDone = 0
        
        currentClass = ""
        for img, class_, fileName in self.load_data("all", grayscale=False, outputActions=False, yieldFilename=True):
            fileId = osPath.splitext(fileName)[0]
            path = normTestDataRootPath + "/" + class_ + "/" 
            # reset counter if new class
            if not currentClass == class_:
                currentClass = class_                
                utils.create_dir_if_necessary(path)
            path += str(numberOfImagesDone) + "_it" + str(iteration)
            
            # calculate the actual number of ops for this image using the numberOfOps range tuple
            numberOfOpsForImage = randint(numberOfOps[0], numberOfOps[1])

            # get a list of unique indices for operations to perform on images if uniqueOps requires it
            ops = []
            if uniqueOps:
                ops = sample(range(len(iterationOps)), numberOfOpsForImage)
            else:
                ops = [randint(0,len(iterationOps)-1) for _ in xrange(numberOfOpsForImage)]
            for op in ops:
                changedImg = None
                # check if op needs a parameter
                if iterationOps[op][1]:
                    parameterIndex = randint(0, len(iterationOps[op][1])-1)
                    changedImg = iterationOps[op][0](img, iterationOps[op][1][parameterIndex])
                else:
                    changedImg = iterationOps[op][0](img)
                changedImgPath = path + "_" + str(numberOfnewImagesDone) + "_OP" + str(op) + ".jpg"
                cv.imwrite(changedImgPath, changedImg)
                numberOfnewImagesDone += 1
            # save original image in new dataset if saveOriginalImage requires it
            if saveOriginalImage:
                cv.imwrite(path + ".jpg", img)
            numberOfImagesDone += 1
            utils.show_progress(True, numberOfImagesDone, numberOfImages, "Iteration {0} - New images: {1}\tProgress:", iteration, numberOfnewImagesDone)
        numberOfTotalImages = numberOfImagesDone + numberOfnewImagesDone
        print "\nIteration {0} done.\nCurrent number of images in data set: {1}\nReloading dataset.".format(iteration, numberOfTotalImages)

        self.reset_data_set()
        loadedSize = self.set_root_path(normTestDataRootPath)
        if not loadedSize >= numberOfTotalImages:
            print "Reloading was not successfull! Number of actual reloaded images: {0} - Expected number of images: {1}.".format(loadedSize, numberOfTotalImages)
            raw_input("Press any key to continue.")
            return None
        print "Reloading successfull."
        return numberOfnewImagesDone

