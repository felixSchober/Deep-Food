import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pickle
import os
import operator
from misc import utils
from classification.classifier import Classifier
from classification.model import ModelType
from classification.svm import OpenCvSVM, SklearnSVM
from classification.kn_neighbors import KNearestNeighbours
from misc.model_tester import ModelTester, get_mean_accuracy
from data_io.model_IO import SvmModelSaver
import data_io.settings as Settings
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import logging


class FeatureClassifier(Classifier):
    """abstract class of a standard classifier"""
    
    def __init__(self, svmType, svmParams, testData=None, name="", grayscale=True, description="", imageSize=16384):
        super(FeatureClassifier, self).__init__(imageSize, name, description, testData, grayscale)
        self.svms = {}
        self.svmType = svmType    
        self.svmParams = svmParams        
        self.validLoss = 1
        self.modelSaver = SvmModelSaver(self.testData.get_root_path(), self.name, self.description)

           
    def __try_load_class_feature_vectors(self, classes):
        """
        tries to reduce the redundant generation of feature vectors for each class
        by loading already computed features.
        This function only returns the vectors if all classes have been computed before.
        
        Keyword arguments:
        classes -- list of classes that should be loaded            
        """

        featureVectors = []

        for class_ in classes:
            path = utils.get_temp_path() + class_ + "_vectors.tmp"

            if not os.path.isfile(path):
                yield (class_, [])
                continue

            # load the vectors
            vectors = []
            with open(path, "r") as f:
                vectors = pickle.load(f) 

            if vectors:
                yield (class_, vectors)
            else:
                yield (class_, [])

    def __save_class_feature_vectors(self, class_, vectors):
        """
        tries to reduce the redundant generation of feature vectors for each class by saving already computed features.
        
        Keyword arguments:
        class_ -- class of feature vectors
        vectors -- vectors to save

        """

        path = utils.get_temp_path() + class_ + "_vectors.tmp"

        # no need to save
        if os.path.isfile(path):
            return

        with open(path, "wb") as f:
            pickle.dump(vectors, f)

    def __delete_temp_class_feature_vectors(self, fileExtension=".tmp"):
        """ Deletes all files in the temp directory with fileExtension."""

        if Settings.G_DETAILED_CONSOLE_OUTPUT:
            print "Deleting temporary class vectors"
        tempPath = utils.get_temp_path()
        filelist = [ f for f in os.listdir(tempPath) if f.endswith(fileExtension) ]
        for f in filelist:
            os.remove(tempPath + f)

    def create_SVM_samples(self):
        """
        yield returns positive and negative feature vectors for one-vs-all SVM training.
        """

        # Creates samples for a one-vs-all SVM.
        # Step 1.0 Clean up possibly existing temp vectors from previous unfinished sessions.
        # Step 1.1: Iterate through all classes.
        # Step 1.2: Create image descriptor using SIFT and the trained BoW for the class -> positive sample vector.
        # Step 1.3: Iterate through all other classes and create descriptors for them -> negative sample vector.
        # Step 2:   Yield return the samples.
        # Next class -> Step 1.2.

        self.__delete_temp_class_feature_vectors()

        if Settings.G_DETAILED_CONSOLE_OUTPUT:
                print "\nCreating SVM samples."

        for class_ in self.testData.classes:
            if Settings.G_DETAILED_CONSOLE_OUTPUT:
                print "Create SVM samples for", class_             
            featureVectors = []
            labels = []

            # **POSITIVE VECTORS**
            # we can not load per computed positive feature vectors because we need the whole range of the samples (all images)
            # we only use a small subset for the negative images not the whole set to keep the number of negative and positive features the same.
            
            # class_ is the positive sample for this round -> get images for it                 
            for img, _ in self.testData.load_data("trainSVM", classes=[class_], grayscale=self.grayscale, outputActions=Settings.G_DETAILED_CONSOLE_OUTPUT, resolutionSize=self.imageSize, transformation=self.transform):
                featureVector = self.create_feature_vector(img)
                if featureVector is None:
                    continue
                labels.append(1)
                featureVectors.append(featureVector)
            positiveFeatures = len(labels)            
            if Settings.G_DETAILED_CONSOLE_OUTPUT:
                print "{0} positive features done.".format(positiveFeatures)

            # **NEGATIVE VECTORS**
            # for the negative data create a list that contains all classes except _class
            negativeClasses = [negClass for negClass in self.testData.classes if not negClass == class_]

            # do we need to create feature vectors or did we already compute them (and can therefore reuse them).
            toCompute = []
            for vectorClass, vector in self.__try_load_class_feature_vectors(negativeClasses):
                # vector was not precomputed before
                if not vector:
                    toCompute.append(vectorClass)
                    continue
                labels.extend([-1 for _ in xrange(len(vector))])
                featureVectors.extend(vector)

            if toCompute:
                # how many images do have to load per class to balance the number of positive features?
                imagesPerClass = int(round(positiveFeatures / len(negativeClasses)))

                # prepare dictionary so that we can save the feature vectors later on
                # structure: {class_: [LIST_OF_FEATURE_VECTORS]}
                classFeatures = {negClass: [] for negClass in toCompute}

                for img, negClass in self.testData.load_data("trainSVM", classes=toCompute, grayscale=self.grayscale, outputActions=Settings.G_DETAILED_CONSOLE_OUTPUT, maxNumberOfImagesPerClass=imagesPerClass, resolutionSize=self.imageSize, transformation=self.transform):
                    labels.append(-1)
                    featureVector = self.create_feature_vector(img)
                    featureVectors.append(featureVector)

                    # for saving later on
                    classFeatures[negClass].append(featureVector)

                #save all features from classFeatures 
                for negClass in classFeatures:
                    self.__save_class_feature_vectors(negClass, classFeatures[negClass])

                if Settings.G_DETAILED_CONSOLE_OUTPUT:
                    print "Negative features done."

            # train one SVM with this
            yield (class_, featureVectors, labels)
        self.__delete_temp_class_feature_vectors()

    def create_feature_vector(self, image):
        raise NotImplementedError("abstract class")

    def create_SVMs(self):
        """
        Creates the actual one-vs-all svms by loading the samples for the svms and calling the train method.
        """
        for (class_, samples, labels) in self.create_SVM_samples():
            if Settings.G_DETAILED_CONSOLE_OUTPUT:
                print "Creating and training SVM for", class_
            svm = None
            if self.svmType == ModelType.OpenCV:
                svm = OpenCvSVM(class_, self.svmParams)            
            elif self.svmType == ModelType.Sklearn:
                svm = SklearnSVM(class_)
            elif self.svmType == ModelType.KNearest:
                svm = KNearestNeighbours(class_)
            svm.train(samples, labels)
            self.svms[class_] = svm
            
            if Settings.G_DETAILED_CONSOLE_OUTPUT:
                print "Creation and training for {0} SVM done.\n**\n".format(class_)
        print "**\n**\nCreation of all {0} SVMs done.".format(self.svmType)

    def predict(self, images, sortResult=True):
        """
        runs the prediction for the image or the images.
        
        Keyword arguments:
        images -- List of images. Note: Out of laziness the method will only predict the first image regardless of the argument length.
        sortResults -- Sort the results based on the response / confidences
        
        Returns:
        list of lists (in case more than one image was provided) of predictions. The first entry is the most likely prediction.
        """
        if not self.trained:
            raise Exception("Classifier is not trained.")


        # make sure the images have the desired size if size is set
        if not self.imageSize is None and self.imageSize != -1: 
            images = [utils.equalize_image_size(image, self.imageSize) for image in images]

        # get input vector for frame.
        inputVectors = [self.create_feature_vector(image) for image in images]
        
        # no keypoints detected
        if inputVectors[0] is None:
            return None
        # get predictions from all SVMs
        predictions = {}
        reverseSorting = False
        for class_, svm in self.svms.iteritems():
            prediction = svm.predict(inputVectors)
            predictions[class_] = prediction
            reverseSorting = svm.reverseSorting

        # order predictions
        if sortResult:
            return sorted(predictions.items(), key=operator.itemgetter(1), reverse=reverseSorting)
        else:
            return predictions

    def save(self):        
        self.modelSaver(self, self.validLoss)  

    def export_evaluation_results(self, results, confMatrices):
        self.tester.export_results_csv(results, confMatrices)
        
    def show_evaluation_results(self, accuracies, confMatrices, header):
        """ Helper method to print the evaluation results to the console."""

        print "Results:"        
        iterations = len(accuracies)

        # average accuracy is only calculated over the testing results which is index 2
        testingAccuracies = [i[2] for i in accuracies]
        averageAccuracy = float(sum(testingAccuracies)) / float(iterations)

        averageLoss = 1 - averageAccuracy
        self.validLoss = averageLoss

        if Settings.G_EVALUATION_DETAIL_HIGH:
            header.append("Confusion score")
            confScores = []
            print "\nIterations:"
            for iteration in xrange(iterations):
                # calculate confusion score
                confScore = self.tester.calculate_confusion_score(confMatrices[iteration])
                confScores.append(confScore)
                
                # apped the confusion score to the table
                accuracies[iteration-1].append(confScore)

                #print "Iteration {0}: Accuracy: {1} - Confusion Score: {2}".format(iteration, accuracies[iteration], confScore)

            print utils.get_table(header, 4, *accuracies)

            averageConfScore = float(sum(confScores)) / float(len(confScores))
            randomScore = self.testData.random_score
            randomScore2 = self.testData.null_score
            print utils.get_table(["avg. acc", "avg. loss", "rdm. acc.", "rdm. acc. impr.", "avg. conf. score"], 3, [averageAccuracy, averageLoss, randomScore, randomScore2, averageConfScore])

            # plot accuracies
            try:
                plt.figure(figsize=(12, 9))
                ax = plt.subplot(111)  
                ax.spines["top"].set_visible(False)  
                ax.spines["right"].set_visible(False) 
                ax.get_xaxis().tick_bottom()  
                ax.get_yaxis().tick_left()  
                plt.xticks(range(1,iterations+1), fontsize=14)  
                plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14) 
                plt.xlabel("Iteration", fontsize=16)  
                plt.ylabel("Accuracy over all classes", fontsize=16)
                plt.title("Results for classifier {0} after {1} iterations".format(self.name, iterations), fontsize=22)  
                plt.axis([0, iterations+1, 0, 1.1])
                x = range(1, iterations+1)
                plt.plot(x, testingAccuracies, 'o', label="results", color=Settings.G_COLOR_PALETTE[0])
                utils.annotate_points(ax, x, testingAccuracies)        
                plt.plot(np.repeat(averageAccuracy, iterations+2), ":", label="avg. accuracy", color=Settings.G_COLOR_PALETTE[1])
                plt.plot(np.repeat(randomScore, iterations+2), "-", label="random accuracy", color=Settings.G_COLOR_PALETTE[2])
                plt.legend()
                plt.show()
                utils.save_plt_figure(plt, "evaluation_results_{0}".format(self.name), path=self.modelSaver.get_save_path())
            except Exception, e:
                logging.exception("Could not plot/show results. Trying to save instead.")                
                utils.save_plt_figure(plt, "evaluation_results_{0}".format(self.name), path=self.modelSaver.get_save_path())
        else:            
            print utils.get_table(header, 4, *accuracies)
            print "Average Accuracy:",averageAccuracy

    def load(self, path):
        
        with open(path + "model", "r") as f:
            self = pickle.load(f)

        # restore SVMs if classifier is trained
        if self.trained and self.svmType != ModelType.Sklearn:
            svmPath = path + "svms/"
            for svmName in self.svms:
                self.svms[svmName].load(svmPath)
        else:
            print "Model is sklearn. Retrain SVMs"
            self.train_and_evaluate()
        return self



