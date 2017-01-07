import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pickle
import operator
from random import randint
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import data_io.settings as Settings
from data_io.testdata import sliding_window
from . import utils
import cv2 as cv
from math import ceil
import logging

def get_mean_accuracy(accuracies):
    # average accuracy is only calculated over the testing results which is index 2
    testingAccuracies = [i[2] for i in accuracies]
    return float(sum(testingAccuracies)) / float(len(accuracies))

class ModelTester(object):    
    """Class to test and evaluate models."""

    def __init__(self, classifier, transformation=None, size=(-1,-1), transformationBack=None):
        """ 
        Instantiates model tester.

        Keyword arguments:
        classifier -- reference to the model.
        transformation -- optional method for image transformation before prediction
        size -- desired image size. Default: (-1, -1) means do not change size
        transformationBack -- optional method for the transformation of the prediction image format back to a displayable format
        """

        self.classifier = classifier
        self.transformation = transformation # method to transform the data (needed for NNs)
        self.transformationBack = transformationBack # since the TestData module applies the transformation we have to reverse the transformation on the images to display them.
        self.size = size

    def __yield_image_predictions(self, segmentIndex, classes=None, maxNumberOfImages=-1, shuffle=False, slidingWindow=False, slidingWindowSize=(300, 300), slidingWindowStride=64):
        """
        Calls the predict method for each image and returns the result of the prediction.

        Keyword arguments:
        segmentsIndex -- Index of the segment to test.
        classes -- List of classes to test. Default: Test all classes
        maxNumberOfImages -- number of images to test. Default: use all
        shuffle -- reshuffle images
        slidingWindow -- test sliding window
        slidingWindowSize -- size of the sliding window. Default: (300, 300) Pixels
        slidingWindowStride -- stride of the sliding window. Default: 64 Pixels

        Returns:
        Generator((class_, prediction, img)) := (Class Name, prediction, image that was tested)
        """

        if classes is None:
            classes = self.classifier.testData.classes

        if shuffle:
            self.classifier.testData.reshuffle_segment_data(segmentIndex)
                       
        prevRandomSamplingStatus = Settings.E_RANDOM_SAMPLE

        for class_ in classes:
            # load test images for this class and predict
            predictions = []
            for img, _ in self.classifier.testData.load_data(segmentIndex, classes=[class_], grayscale=self.classifier.grayscale, outputActions=False, maxNumberOfImagesPerClass=maxNumberOfImages, size=self.size, transformation=self.transformation, resolutionSize=self.classifier.imageSize):
                # classifier tester expects a list in the form of [(class_, [predictions])]
                if slidingWindow:

                    # prevent random sampling                    
                    Settings.E_RANDOM_SAMPLE = False

                    voteDict = {cls: 0 for cls in classes}
                    slVis = np.copy(img)

                    # is slVis grayscale?
                    if self.classifier.grayscale:
                        slVis = cv.cvtColor(slVis, cv.COLOR_GRAY2BGR)

                    for roi, slImg in sliding_window(img, slidingWindowSize, slidingWindowStride):
                        
                        p = self.classifier.predict([slImg])
                        if p is None:
                            continue

                        # outputs the class with highest confidence
                        p = p[0][0]

                        voteDict[p] += 1
                        
                        # overlay imagePart if correct class
                        if p == class_:
                            slVis = roi.overlay_rectangle(slVis)
                    cv.imwrite(self.classifier.modelSaver.get_save_path_for_visualizations() + "slidingWindow/{0}.jpg".format(class_), slVis)
                    print "Sliding Window prediction for class {0} Votes:\n{1}\n\n".format(class_, voteDict)
                    Settings.E_RANDOM_SAMPLE = prevRandomSamplingStatus

                prediction = self.classifier.predict([img])
                if prediction is None:
                    continue
                yield (class_, prediction, img)

    def __yield_class_predictions(self, segmentIndex):
        """
        Calls the predict method for each class and yields the result as a tuple with the class and a list of predictions.

        Keyword arguments:
        segmentIndex -- index of the test data segment

        Returns:
        Generator((class_, predictions)) := (Class name, List of predictions)
        """

        for class_ in self.classifier.testData.classes:
            # load test images for this class and predict
            predictions = [p for _, p, _ in self.__yield_image_predictions(segmentIndex, [class_])]            
            yield (class_, predictions)

    def test_classifier(self, segments=["test"]):
        """
        Completely evaluates a classifier and prints the results to the console window and saves the results to the model directory.

        Keyword arguments:
        segments -- List of segments to test onto

        Returns:
        dictionary of results of the segments.
        """

        if Settings.G_DETAILED_CONSOLE_OUTPUT:
            print "## Testing classifier:\n"
        results = {}
        for segment in segments:
            print "#  Testing",segment
            # stores classes as key and the prediction results (list) as value.
            segmentResults = {}
            precisionRecallValues = {}

            for class_, predictions in self.__yield_class_predictions(segment):

                # number of matches for 1,2,...,numberOfClasses-1 candidates
                topNMatches = [0] * (self.classifier.testData.numberOfClasses - 1)
                images = 0.0
                # load images and predict.
                for prediction in predictions:
                    predictionRank = self.__test_top_n_prediction(class_, prediction)

                    #prevent problems with less than 6 classes
                    maxRank = min(self.classifier.testData.numberOfClasses - 1, len(predictionRank)-1)

                    for i in xrange(maxRank+1):
                        topNMatches[i] += predictionRank[i]
                    images += 1.0

                # Calculate accuracy for class.
                segmentResults[class_] = [matches / images for matches in topNMatches]

                # calculate Precision recall
                precisionValues = []
                recallValues = []
                f1Scores = []
                for top in xrange(self.classifier.testData.numberOfClasses - 1):
                    
                    # All correctly classified items
                    truePositives = float(topNMatches[top])

                    # all predicted images without the correctly predicted images. In case of top-1 the total ammount of images is exactly the number of returned predictions. 
                    # For top-2 we have twice as much predictions to consider.
                    falsePositives = float((len(predictions) * (top+1))-truePositives)

                    # All items that were not correctly classified.
                    falseNegatives = float(len(predictions) - truePositives)

                    precision = truePositives / (truePositives + falsePositives)
                    recall = truePositives / (truePositives + falseNegatives)
                    #f1Score = 2.0 * ((precision * recall) / (precision + recall))
                    precisionValues.append(precision)
                    recallValues.append(recall)
                    #f1Scores.append(f1Score)
                
                precisionRecallValues[class_] = (precisionValues, recallValues)
                if Settings.G_DETAILED_CONSOLE_OUTPUT:
                   print "\t- Testing {0} - Accuracy: {1:.2f}% - T5 Precision: {2:.2f} - T5 Recall: {3:.2f}".format(class_, segmentResults[class_][0]*100, precisionValues[4], recallValues[4])

            # Calculate overall top 1 accuracy.
            segmentAccuracy = sum([a[0] for (_, a) in segmentResults.iteritems()]) / len(segmentResults)
            segmentError = 1 - segmentAccuracy

            # sort accuracies of classes so that we can get the best and worst classes 
            segmentResultsList = segmentResults.items()
            # segmentResultsList contains the top-n accuracies but we only need the top-1 accuracy
            segmentResultsList = [(class_, values[0]) for (class_, values) in segmentResultsList]
            segmentResultsList = sorted(segmentResultsList, key=operator.itemgetter(1), reverse=True)
            # prevent overflow
            bestBound = min(2, len(segmentResultsList))
            worstBound = max(2, len(segmentResultsList)-2)
            bestClasses = segmentResultsList[0:bestBound]
            worstClasses = segmentResultsList[worstBound:]               
            
            results[segment] = [segmentAccuracy, segmentError, bestClasses, worstClasses, segmentResults, precisionRecallValues]         

        # Save the results
        self.save_results(results, False)

        return results

    def plot_random_predictions(self, segmentIndex="test", cols=4):
        """ Creates an image with predictions of random images from the segment index and the model confidences."""

        # result will have a format like this: [(real class, [(class, prediction for class), (class, prediction for class), ...], image)]
        results = []
        for class_, prediction, image in self.__yield_image_predictions(segmentIndex, maxNumberOfImages=1, shuffle=True, slidingWindow=True):
            # convert image back to cv format if neccessary
            if not self.transformationBack is None:
                image = self.transformationBack(image)
            # take the first 4 predictions and turn them to percent (looks better)
            top4 = [(cls, p[0]*100.0) for cls, p in prediction[0:4]]
            top4.reverse()
            # convert the images from bgr to rgb if color
            if len(image.shape) > 2 and image.shape[2] != 1:
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results.append((class_, top4, image))
            
        
        # plot results
        rows = int((ceil(float(len(results)) / cols)) * 2)
        f, axarr = plt.subplots(rows, cols)
        f.set_size_inches(int(cols*4),int((rows/2)*5))
        f.suptitle(str(self.classifier), fontsize=20)
        i = 0
        for y in range(0, rows, 2):
            for x in range(cols):
                if i >= len(results):
                    # disable axis for empty images
                    axarr[y, x].axis('off')
                    axarr[y+1, x].axis('off')
                    continue
                
                if self.classifier.grayscale:
                    axarr[y, x].imshow(results[i][2], cmap = cm.Greys_r)
                else:
                    axarr[y, x].imshow(results[i][2])
                axarr[y, x].set_title(results[i][0])
                axarr[y, x].axis('off')

                # add bars
                barPlotValues = zip(*results[i][1]) # barPlotValues[0] = labels, barPlotValues[1] = prediction values
                positions = np.arange(0, 0.8, 0.2)
                bars = axarr[y+1, x].barh(positions, barPlotValues[1], align='center', height=0.2, color=Settings.G_COLOR_PALETTE[0], linewidth=0)

                # color bar of correct result differently
                if results[i][0] in barPlotValues[0]:
                    correctBarIndex = barPlotValues[0].index(results[i][0])
                    bars[correctBarIndex].set_color(Settings.G_COLOR_PALETTE[3])

                for class_, yPos in zip(barPlotValues[0], positions):
                    axarr[y+1, x].text(95, yPos, class_, horizontalalignment='right', verticalalignment='center', fontsize=8)
                axarr[y+1, x].axis('off')
                axarr[y+1, x].set_xlim([0, 100])
                i += 1
        name = "RandomResults_" + self.classifier.name + "_" + utils.get_uuid()
        utils.save_plt_figure(plt, name, self.classifier.modelSaver.get_save_path_for_visualizations())
        raw_input("Random results plotting complete. Press any key to continue")

    def __test_top_n_prediction(self, class_, predictions):
        """ Computes the top-N predictions."""

        topN = []
        for n in range(1, len(predictions)):

            # take n-size slice out of predictions and create list without the confidence.
            # the result should look something like this for the top 3 ["bread", "chocolate", "rice"] if the list looks like this
            # ["bread", "chocolate", "rice", "noodles", "ice", ...].
            topNPredictions = [c for (c, _) in predictions[:n]]

            if class_ in topNPredictions:
                topN.append(1)
            else:
                topN.append(0)

        return topN

    def yield_top_n_results_as_list(self, results):
        """ Returns a generator that yields the top-N results."""

        for segment in results:
            result = results[segment]

            # Iterate through classPredictions and display the top-n categories 
            for class_ in result[4]:
                classPredictions = result[4][class_]
                topN = []      
                for accuracy in classPredictions:                    
                    topN.append(accuracy)
                yield (segment, class_, topN)

    def format_results_string(self, results):
        """ Formats the results and creates a string that can be saved or printed to the console."""

        output = ""
        #overallAccuracy, classPredictions = results
        output += "\n\n\nTest report for " + self.classifier.name + "\n"
        detailedOutput = "\n\nDetailed report:"
        outputRows = []
        for segment in results:
            result = results[segment]
            outputRows.append([segment, result[1], result[0], result[2], result[3]])

            #detailed output:
            detailedOutput += "\n\n********************************************************\nSegment " + segment + "\n"
            
            detailedOutputRows = []
            # Iterate through classPredictions and display the top5 categories 
            for class_ in result[4]:
                classPredictions = result[4][class_]
                detailedRow = [class_]      
                for accuracy in classPredictions:                    
                    detailedRow.append(accuracy)
                detailedOutputRows.append(detailedRow)
            detailedOutputTitle = ["class"]
            detailedOutputTitle.extend(self.__get_top_title())
            detailedOutput += utils.get_table(detailedOutputTitle, 6, *detailedOutputRows).get_string()

        output += utils.get_table(["segment", "segment_loss", "segment_accuracy", "top-2", "flop-2"], 6, *outputRows).get_string()
        output += detailedOutput

        return output

    def __get_top_title(self):
        """ Returns the Top-N title used for the csv output."""

        return ["Top " + str(n+1) for n in range(self.classifier.testData.numberOfClasses-1)]

    def export_results_csv(self, results, confMatrices):
        """ Exports the results to a csv file."""

        writer = self.classifier.modelSaver.get_csv_exporter()
        
        # export test data stats
        writer.export(self.classifier.testData.export_test_data_information(), "testDataStats")

        # get mean / std images if pre computed
        mean = self.classifier.testData.mean_image
        if not mean is None:
            # there is propably also a std image
            std = self.classifier.testData.std_image
            cv.imwrite(self.classifier.modelSaver.get_save_path_for_visualizations() + "testDataMeanImage.jpg", mean)
            cv.imwrite(self.classifier.modelSaver.get_save_path_for_visualizations() + "testDataStdImage.jpg", std)

        # export conf matrices and results
        iterationOutput = []
        iterationOutputTitle = ["iteration", "segment", "segment loss", "segment accuracy"]
        iterationOutputTitle.extend([class_ + " t1 accuracy" for class_ in self.classifier.testData.classes])
        iterationOutput.append(iterationOutputTitle)

        for iteration in xrange(len(results)):
            if iteration < len(confMatrices):
                self.export_confusion_matrix_as_csv(confMatrices[iteration], fileName="ConfusionMatrix_iteration" + str(iteration+1))
            try:
                iterationResults = results[iteration]
            except:
                # could not extract iterationResults because in this case results does not contain a list of iterations because it had only one iteration.
                # This shouldn't happen -> FIXME
                return
            for segment in iterationResults:
                result = iterationResults[segment]
                iterationOutputRow = [iteration+1, segment, result[1], result[0]]
                for class_ in self.classifier.testData.classes:
                    iterationOutputRow.append(result[4][class_][0])
                iterationOutput.append(iterationOutputRow)

                # export precision recall
                precisionRecallValues = result[5] # precisionRecallValues[class_] = (precisionValues, recallValues)
                
                for class_ in precisionRecallValues:
                    precisionCSV = [["Top-N", "precision", "recall"]]
                    precisionValues, recallValues = precisionRecallValues[class_]
                    for i in xrange(len(precisionValues)):
                        precisionCSV.append([i+1, precisionValues[i], recallValues[i]])
                    writer.export(precisionCSV, "{0}_PrecisionRecall_{1}".format(segment, class_))



            # export top-n results
            segmentTopResults = []
            segmentTopResultsTitle = ["segment", "class"]
            segmentTopResultsTitle.extend(self.__get_top_title())
            segmentTopResults.append(segmentTopResultsTitle)
            for (sgmt, class_, topN) in self.yield_top_n_results_as_list(iterationResults):
                segmentTopResultsRow = [sgmt, class_]
                segmentTopResultsRow.extend(topN)
                segmentTopResults.append(segmentTopResultsRow)
            writer.export(segmentTopResults, name="iteration_" + str(iteration+1) + "_topN")
        writer.export(iterationOutput, name="detailedResults")
        
    def save_results(self, results, exportToCSV=True):
        """ Exports the result string to a text file and saves the results to csv if exportToCSV is True."""
        
        path = self.classifier.modelSaver.get_save_path()

        resultString = self.format_results_string(results)
        with open(path + "Results.txt", "w") as f:
            f.write(resultString)

        if exportToCSV:
            self.export_results_csv(results, [])

    def plot_confusion_matrix(self, save=True, show=True, confMatrix=None):
        """ 
        Plots a confusion matrix and saves the image.

        Keyword arguments:
        save -- Save confusion matrix
        show -- Show confusion matrix. Only works locally or via vcn.
        confMatrix -- precomputed confusion matrix - Default: Compute new.
        """

        if confMatrix is None:        
            confMatrix = self.compute_confusion_matrix()

        # normalize matrix
        normConfMatrix = []
        for i in confMatrix:
            a = sum(i, 0)
            temp = []
            for j in i:
                temp.append(float(j)/float(a))
            normConfMatrix.append(temp)

        # can we plot labels? Only plot labels if we have less than 10 classes
        showLables = len(confMatrix[0]) < 10


        # we can not create the figure on the server since tkinter does not work because the server does not have a display output.
        # in this case we save the confusion matrix which we can load on a machine with a display to create the plot from there.
        try:
            # create figure and clear it
            fig = plt.figure()
            plt.clf()
            ax = fig.add_subplot(111)
            ax.set_aspect(1)
            res = ax.imshow(np.array(normConfMatrix), cmap=plt.cm.jet, interpolation='nearest')

            if showLables:
                w = len(confMatrix)
                h = len(confMatrix[0])
                for x in xrange(w):
                    for y in xrange(h):
                        if normConfMatrix[x][y] > 0:
                            ax.annotate(str(confMatrix[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')
                plt.xticks(range(w), self.classifier.testData.classes)
                plt.yticks(range(h), self.classifier.testData.classes)
            else:
                plt.xticks([]),plt.yticks([])
            cb = fig.colorbar(res)
                        
            if show:
                plt.show()

            if save:
                utils.save_plt_figure(plt, "conf_matrix_{0}".format(self.classifier.name))
            
        except Exception, e:            
            path = utils.get_temp_path() + "ConfMatrix.tmp"
            logging.exception("Error while saving confusion matrix. Saving results in {0}.".format(path))
            self.export_confusion_matrix_as_csv(confMatrix)
    
    def calculate_confusion_score(self, confMatrix=None):
        """ 
        Calculates the sum the the diagonal of the confusion matrix.
        This is the number of correctly classified images.
        """
        if confMatrix is None:
            confMatrix = self.compute_confusion_matrix()

        diagonalSum = np.trace(confMatrix)
        return diagonalSum

    def export_confusion_matrix_as_csv(self, confMatrix=None, fileName="ConfusionMatrix"):
        """
        Exports the confusion matrix to csv.

        Keyword arguments:
        confMatrix -- precomputed confusion matrix
        """

        if confMatrix is None:
            confMatrix = self.compute_confusion_matrix()

        writer = self.classifier.modelSaver.get_csv_exporter()
        writer.export(confMatrix, fileName)

        # export keys
        convKeys = [range(self.classifier.testData.numberOfClasses)]
        convKeys.append(self.classifier.testData.classes)
        writer.export(convKeys, fileName + "_Keys")

    def compute_confusion_matrix(self, export=True):
        """ Computes the confusion matrix for the classifier using the test segmentindex. """

        # construct the confusion matrix
        confusionMatrix = np.zeros((self.classifier.testData.numberOfClasses, self.classifier.testData.numberOfClasses))
        classes = self.classifier.testData.classes
        classIndex = 0
        for class_, predictions in self.__yield_class_predictions("test"):
            for prediction in predictions:
                predictedClass, _ = prediction[0]
                confusionMatrix[classIndex][classes.index(predictedClass)] += 1
            classIndex += 1
        if export:
            self.export_confusion_matrix_as_csv(confusionMatrix)
        return confusionMatrix

    def classify_image_folder(self, path):
        """ Classifies images from a folder from a given path and prints the top-1 prediction on the console."""

        if not path.endswith("/"):
            path += "/"

        if not utils.check_if_dir_exists(path):
            raise Exception("Path '{0}' does not exist.".format(path))

        from os import walk

        # Load filenames
        _, _, filenames = walk(path).next()

        # Load images
        #Load flag for cv.imread.
        loadFlag = cv.IMREAD_GRAYSCALE if self.classifier.grayscale else cv.IMREAD_UNCHANGED

        from data_io.testdata import load_image

        for imgName in filenames:
            imgPath = path + imgName
            img = load_image(imgPath, loadFlag, 1)
            if self.size != (-1, -1):
                img = utils.crop_to_square(img)
                desiredArea = self.size[0] * self.size[1]
                img = utils.equalize_image_size(img, desiredArea)
            if not self.transformation is None:
                img = self.transformation(img)
            
            prediction = self.classifier.predict([img]) 
            print "Img {0}: {1}".format(imgName, prediction[0])# only top-1 prediction

    def classify_webcam(self):
        """ Classifies frames from the webcam."""

        cam = cv.VideoCapture(0)
        while True:
            ret_val, img = cam.read()
            cv.imshow('TUM FoodCam', img)
            try:
                prediction = self.classifier.predict([img]) 
                print "{0}".format(prediction[0])# only top-1 prediction
            except:
                pass
            if cv.waitKey(1) == 27: 
                break  # esc to quit

        cv.destroyAllWindows()

