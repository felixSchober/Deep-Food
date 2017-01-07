import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pickle
import os
import cv2 as cv
import numpy as np
from math import ceil
from matplotlib import pyplot as plt
from segmentation.image_region import ImageRegion
from segmentation.rectangle import Rectangle
import misc.utils as utils
from data_io.testdata import load_image
from __builtin__ import super
from random import randint
import operator

def nothing(x):
    pass

class ImageSegmentation(object):
    """ A class which provides basic object separation methods."""

    def __init__(self):
        self.originalImage = None
        self.mealRegions = []


    def apply_canny(self, var1, var2, image):
        img = image.copy()
        return cv.Canny(img, var1,var2)
        #cv.imshow("CannyTest", img)



    def apply_edge_detection(self, image, iterations=2, showSteps=True, imageConversion=cv.COLOR_BGR2RGB):
        """ 
        Applies image enhancements and edge detection.

        Keyword arguments:
        image -- image to work on
        iterations -- number of canny and dilation operations
        showSteps -- show the results of the iterations afterwards
        imageConversion -- image channel conversion for showSteps
        """

        visualization = ([], [])
        self.__append_to_visualization_list(visualization, cv.cvtColor(image, imageConversion), "Original", showSteps)

        for i in xrange(iterations):
            # Test edge detection parameters
            #cv.namedWindow("CannyTest")
            #cv.createTrackbar("low_threshold", "CannyTest", 0, 200, nothing)
            #cv.createTrackbar("high_threshold", "CannyTest", 0, 200, nothing)
            #orgGrayImg = image.copy()
            #low = 0
            #high = 0
            #while(True):
            #    cv.imshow("CannyTest", image)
            #    k = cv.waitKey(1) & 0xFF
            #    if k == 27:
            #        break
            #    low = cv.getTrackbarPos("low_threshold", "CannyTest")
            #    high = cv.getTrackbarPos("high_threshold", "CannyTest")
            #    image = self.apply_canny(low, high, orgGrayImg)
            #cv.destroyAllWindows()

            image = cv.Canny(image, 150,50)
            self.__append_to_visualization_list(visualization, image, "Iteration " + str(i) + " Canny", showSteps)
        
            # Dilate the edge detected lines so that the contour algorithm performs better.
            image = utils.dilate_image(image, iterations=1)
            self.__append_to_visualization_list(visualization, image, "Iteration " + str(i) + " Dilate", showSteps)

        if showSteps:
            rows = ceil(float(len(visualization[1])) / 5)
            plt.figure(1, (20, 20))
            for i in xrange(len(visualization[1])):
                plt.subplot(rows,5,i+1),plt.imshow(visualization[0][i],'gray')
                plt.title(visualization[1][i])
                plt.xticks([]),plt.yticks([])
            plt.show()
            utils.save_plt_figure(plt, "image_segments_edges_steps")

        return image

    def __append_to_visualization_list(self, lists, image, title, shouldAppend):
        """ 
        Helper function for cleaner look: appends and copies an image to a list if shouldAppend is true.
        """
        if shouldAppend:
            lists[0].append(np.copy(image))
            lists[1].append(title)

    def __find_largest_contour(self, image, areaThreshold):
        """ 
        Finds the largest contour in the image given an image and an areaThreshold to filter smaller areas.
        """

        cnts, hier = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if len(cnts) == 0:
            # no new contours were found (unlikely)
            return (None, None)

        # Search for the largest contour.
        # Calculate all areas and select the largest
        cntAreas = [cv.contourArea(a) for a in cnts]
        largestCnt = max(cntAreas)
        # stop if contour is to small
        if (largestCnt < areaThreshold):
            return (None, None)
        largestCntAreaIndex = cntAreas.index(largestCnt)

        return (cntAreas[largestCntAreaIndex], cnts[largestCntAreaIndex])

    def __create_image_region(self, contour):
        """ Returns an imageRegion from a contour."""

        return ImageRegion(contour)
    
    def yield_image_objects(self, image, areaThreshold=50, maskImage=True):
        """ 
        Iterates through contours and yields them.
        """
        # generate local copy of image.
        
        tempImage = np.copy(image)
        
        blobNumber = 0
        while True:
            # find largest contour in (remaining) image            
            area, cnt = self.__find_largest_contour(tempImage, areaThreshold)

            

            # no remaining contour left.
            if area is None or cnt is None:
                break

            # get the object (region of interest) as an image only from the original image
            roi = self.__create_image_region(cnt)
            roiImg = self.originalImage
            if maskImage:
                roiImg = roi.get_roi_image(self.originalImage)
            roiImg = roi.crop_image_region(roiImg)
            blobNumber += 1            

            # delete contour from image for the next iteration by painting the area black
            cv.drawContours(tempImage, [cnt], 0, (0,0,0), thickness=-1)
            try:
                yield (roi, roiImg, area)
            except GeneratorExit:
                break
          
        tempImage = None  

class ObjectSegmentation(ImageSegmentation):
    """This class searches for the meal and the marker and calculates (based on the marker) the area of the meal."""

    # Simple chessboard pattern:
    # w b
    # b w
    # The pattern actually has 6 x 5 squares, but has 5 x 4 = 20 'ENCLOSED' corners
    MARKER_CHESSBOARD_PATTERN = (5,4)

    # Real world marker size in cm
    MARKER_SIZE = (8,6)

    # Custom marker keypoints and descriptors (None, None) if no custom marker
    CUSTOM_MARKER = (None, None)

    CUSTOM_MARKER_IMAGE = None

    def __init__(self):
        super(ObjectSegmentation, self).__init__()
                
        self.markerRegion = None
        self.markerCorners = []
        self.restore_custom_marker()

    def learn_custom_marker(self):
        """ "Learns a custom rectangular marker object by calculating SIFT keypoints."""

        markerImage = load_image(utils.get_path_via_file_ui(), 0, 1)
        # save image 
        cv.imwrite(utils.get_data_path() + "segmentationMarkerImage.jpg", markerImage)
        sift = cv.SIFT()
        self.CUSTOM_MARKER = sift.detectAndCompute(markerImage, None) # contains keypoints and descriptors
        self.MARKER_SIZE = (utils.value_question("[Marker Size]", "What is the marker width in cm?", "f"), utils.value_question("[Marker Size]", "What is the marker height in cm?", "f"))

        # save marker (not saving keypoints as they are not pickable)
        markerFile = {"marker": self.CUSTOM_MARKER[1], "markerDimension": self.MARKER_SIZE}
        with open(utils.get_data_path() + "segmentationMarker", "wb") as f:
            pickle.dump(markerFile, f)

        raw_input("Learning complete. Press any key to continue")

    def restore_custom_marker(self):
        """ Restores a custom marker from a file."""

        path = utils.get_data_path() + "segmentationMarker"
        if utils.check_if_file_exists(path):  
            with open(path, "r") as f:
                markerFile = pickle.load(f) 
                self.MARKER_SIZE = markerFile["markerDimension"]
                

                # restore kps
                self.CUSTOM_MARKER_IMAGE = cv.imread(utils.get_data_path() + "segmentationMarkerImage.jpg", 0)
                sift = cv.SIFT()
                kp = sift.detect(self.CUSTOM_MARKER_IMAGE,None)
                self.CUSTOM_MARKER = (kp, markerFile["marker"])

                print "restored custom marker"

    def process_image(self, image):
        """ 
        Tries to segment an image with edge detection and searches for a marker.
        If a marker was found the area (in cm sq) is calculated.
        Call visulize_regions for a visualization of the results.
        """

        if image is None:
            raise AttributeError("Image is None")

        self.originalImage = image

        #self.originalImage = cv.fastNlMeansDenoising(image)

        edgeImage = self.apply_edge_detection(image, showSteps=True)
        
        self.__find_meal_and_marker_regions(edgeImage)

    def visulize_regions(self):
        """ Visualizes the results generated by process_image."""
        
        if self.originalImage is None:
            raise AttributeError("originalImage is not initialized. Run process_image() first")

        # Convert image to back to color to enable colored markers
        #tempImage = cv.cvtColor(self.originalImage, cv.COLOR_GRAY2BGR)
        tempImage = np.copy(self.originalImage)
        

        # mark marker
        if self.markerRegion is not None:
            cv.rectangle(tempImage,self.markerRegion.upperLeft,self.markerRegion.lowerRight,(0,0,255),2)

            # bad: this will fail if Custom marker
            try:
                cv.drawChessboardCorners(tempImage, ObjectSegmentation.MARKER_CHESSBOARD_PATTERN, self.markerCorners, True)
            except:
                # nothing to do. Fine if custom marker
                pass

            # show areas as text            
            marker_iA, _, _ = self.__calculate_marker_image_dimension()
            marker_rA = self.calculate_marker_real_area()
            meal_iA = self.calculate_meal_image_area()
            meal_rA = self.calculate_meal_real_area(meal_iA, marker_iA)            
            markerAreaText = "Marker Area: " + str(round(marker_iA)) + "Px^2 - " + str(round(marker_rA)) + "cm^2"   
            mealAreaText = "Meal Area: " + str(round(meal_iA)) + "Px^2 - " + str(round(meal_rA)) + "cm^2"    
            print markerAreaText    
            print mealAreaText
        else:
            #no marker was found:
            markerAreaText = "No marker found"
            mealAreaText = ""
        
        font = cv.FONT_HERSHEY_DUPLEX
        fontSize = 1
        cv.putText(tempImage, markerAreaText, (10,25), font, fontSize, (0,0,255), 2)
        cv.putText(tempImage, mealAreaText, (10,55), font, fontSize, (0,255,0), 2)
        
        #mark food        
        for region in self.mealRegions:
            cv.rectangle(tempImage,region.upperLeft,region.lowerRight,(0,255,0),2)
        
        return tempImage

    def get_meal_image(self):
        """ TODO
        """
        return self.mealRegions[0].crop_image_region(self.originalImage)

    def __find_meal_and_marker_regions(self, image):
        """ 
        Iterates through contours and tries to find the meal and the marker.
        When the marker is found we remove it and run the search for the meal again but without the marker this time.
        This ensures that the marker is prioritized over the meal if the contours are connected as it happened in [3].
        """

        

        while not self.mealRegions:
            workingImage = None # prevent memory leaks
            workingImage = np.copy(image)
            for roi, roiImg, _ in self.yield_image_objects(workingImage, maskImage=False):
                # is ROI the marker?
                isMarker, corners = self.__is_marker(roiImg)
                if isMarker:
                    self.markerRegion, self.markerCorners  = self.__refine_marker_region(corners, roi)
                    image = self.__remove_marker_from_image(image)
                    break
                else:
                    if not self.mealRegions:
                        self.mealRegions.append(roi)

                    if self.markerRegion is None:
                        # Keep on searching until we find the marker
                        continue
                    else:
                        break

    def __is_marker(self, image):
        """ 
        Tries to find the marker for the given image extract.
        Returns True, [corners] if marker is found and False, None if the image does not contain a marker.
        """
        # find custom marker first
        
        try:
            _, corners = cv.findChessboardCorners(image, ObjectSegmentation.MARKER_CHESSBOARD_PATTERN)
        except:
            return False, None
        
        if corners is not None and len(corners) > 0:
            return (True, corners)

        if self.CUSTOM_MARKER_IMAGE == None:
            return (False, None)

        # Custom marker -> match it
        sift = cv.SIFT()
        imgKp, imgDesc = sift.detectAndCompute(image, None)
        markerKp = self.CUSTOM_MARKER[0]
        markerDesc = self.CUSTOM_MARKER[1]

        bf = cv.BFMatcher()
        matches = bf.knnMatch(imgDesc,markerDesc, k=2)
        # Apply ratio test
        good = []
        matchCounter = 0
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                matchCounter += 1
                good.append([m])
        # prevent zero division if no matches
        if len(matches) == 0:
            matchRatio = 0.0
        else:
            matchRatio = float(matchCounter) / float(len(matches))
        if matchRatio  > 0.45 and len(good) > 8:
            img3 = utils.visulize_matches(good, markerKp, imgKp, self.CUSTOM_MARKER_IMAGE, image)
            imgKpV = cv.drawKeypoints(image, imgKp)
            cv.imwrite("C:/Users/felix/Desktop/Segmentation/results/resultsMarker_matches.jpg", img3)
            #cv.imshow("Matches", img3)#
            cv.imwrite("C:/Users/felix/Desktop/Segmentation/results/resultsMarker_keypoints.jpg", imgKpV)
            #cv.imshow("Keypoints", imgKpV)

            #cv.waitKey(0)


            # extract new keypoints with new sift (as much as we can to get the positions)
            sift = cv.SIFT(contrastThreshold=0.002)
            imgKp = sift.detect(image)
            points = np.array([[[int(k.pt[0]), int(k.pt[1])]] for k in imgKp])

            return (True, points)        
        return (False, None)

    def __refine_marker_region(self, markerCorners, markerRegion):
        """
        Refines the marker region by reducing the image region to the recognized corners. Needed because of experiment [3].
        
        Keyword arguments:
        markerCorners -- list of the corners that define the marker.
        markerRegion -- the image region of the marker
        """
        # calculate displacement of corners because corners were calculated for the cropped image region
        xOffset, yOffset = markerRegion.upperLeft
        
        for i in xrange(len(markerCorners)):
            corner = markerCorners[i][0]
            corner[0] += xOffset
            corner[1] += yOffset

        markerBoundingRect = cv.minAreaRect(markerCorners)

        if self.CUSTOM_MARKER == (None, None):
            markerBoundingRect = self.__fit_marker_boundingRect_size_to_actual_marker(markerBoundingRect)

        # convert the box2D structure of markerBoundingRect to a more useful rectangle structure with 4 corner points.
        # this rectangle structure acts as a contour and can be used by drawContours
        markerBoundingBox = np.int0(cv.cv.BoxPoints(markerBoundingRect))
        
        return (ImageRegion(contour=markerBoundingBox, cutmask=markerRegion.cutmask), markerCorners)
    
    def __fit_marker_boundingRect_size_to_actual_marker(self, markerBoundingRect):
        """ 
        When the marker is recognized through is_marker the corners only contain the inner region of the marker. Experiment [3]
        To get the whole marker area this function increases the size by exactly one square.

        Keyword arguments:
        markerBoundingRect -- a bounded / fitted rectangle around the marker

        Returns a Box2D structure: (center(x,y), (width, height), angle of rotation)
        """

        # get marker size (w, h) so that we know how much we have to increase the size
        _, w, h = self.__calculate_marker_image_dimension(markerBoundingRect)

        # get "one square"
        oneSquare = w / ObjectSegmentation.MARKER_CHESSBOARD_PATTERN[0]

        # adjust the bounding rectangle + some padding
        w += 4 * oneSquare + 0.25 * w
        h += 4 * oneSquare + 0.25 * h

        # angle and center remain unchanged        
        return (markerBoundingRect[0], (w, h), markerBoundingRect[2])             

    def __remove_marker_from_image(self, image):
        """ 
        Takes the source image and deletes the marker from it.
        In a few cases the meal and the marker contour get mixed.
        For this case we delete the marker region from the image so that the meal region can be found.
        """

        # create (positive) mask for the marker and subtract it from the image thus removing it.
        removedMarker = image - self.markerRegion.create_mask(image.shape)
        
        # normalize to 1
        removedMarker = removedMarker/(removedMarker.max()/1)

        return removedMarker

    def __calculate_marker_image_dimension(self, markerBoundingRect=None):
        """ 
        Calculates the marker dimensions using the OpenCV function minAreaRect
        which tries to fit a convex-hull-rectangle over a given set o points.
        In this case these points are the points is_marker identified.

        Keyword arguments:
        markerBoundingRect -- a bounded / fitted rectangle around the marker
        """
        
        if markerBoundingRect is None:      
            t1, (w, h), t2 = cv.minAreaRect(self.markerCorners)
        else:
            _, (w, h), _ = markerBoundingRect

        # area approximation (because it isn't an exact rectangle)
        a = w * h
        print "Marker: W: {0} x H: {1} - A: {2}".format(w,h,a)
        return (a, w, h)

    def calculate_marker_real_area(self):
        """ Calculates the markers real size in cm sq."""

        return ObjectSegmentation.MARKER_SIZE[0] * ObjectSegmentation.MARKER_SIZE[1]

    def calculate_meal_image_area(self, mealRegion=None):
        """ 
        Calculates the pixel area of the meal for all detected meal regions. (default)
        Calculates the pixel area of a meal provides through argument mealRegion.
        """
        if mealRegion is not None:
            return cv.contourArea(mealRegion.contour)

        return sum([cv.contourArea(region.contour) for region in self.mealRegions])

    def calculate_meal_real_area(self, mealImageArea=None, markerImageArea=None, mealRegion=None):
        """ 
        Calculates the real meal area in cm sq.
        If the marker size has been pre calculated these values will be taken.
        Otherwise this function will calculate them for you.
        If the mealRegion is provided this function will return the area for the provided mealRegion instead
        """

        # check if marker has been found
        if self.markerRegion is None:
            return -1

        if mealImageArea is None:
            mealImageArea = self.calculate_meal_image_area(mealRegion=mealRegion)
        if markerImageArea is None:
            markerImageArea, _, _ = self.__calculate_marker_image_dimension()

        x = self.calculate_marker_real_area() / markerImageArea
        a = x * mealImageArea
        return a

class ColorSegmentation(ImageSegmentation):
    """ Class for color segmentation of meal parts."""

    # deprecated
    COLOR_WINDOW_SIZE = (20,20)


    def __init__(self):
        super(ColorSegmentation, self).__init__()
        self.mealImages = []
        self.colorMap = None

    def process_image(self, image):
        """ 
        Tries to segment an image using color filtering
        If a marker was found the area (in cm sq) is calculated.
        Call visulize_regions for a visualization of the results.
        """
        self.originalImage = image
        self.originalImage = cv.resize(image, (0,0), fx=0.5, fy=0.5)
        self.colorMap = np.zeros_like(image)
        
        # 2. go through meal image build a list of common colors
        #colors = self.__calculate_mean_color_list()
        #plt.imshow(self.colorMap, 'gray')
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #utils.get_improved_median_cut(image)
        print "Calculating median cut"
        colors = utils.get_modified_median_cut_palette(image)

        #colors = []
        #with open(os.getcwd() + "\\\\Test.txt", "r") as f:
        #    colors = pickle.load(f)
        
        # 3. Filter the image by the colors generated
        masks = self.__color_filter_image(colors, visualize=True)
        
        # 4. sort the masks and eliminate black masks
        masks = self.__sort_image_masks(masks, visualize=False)

        if not masks:
            return

        # 5. find the objects
        self.mealImages = self.__find_objects_of_masks(masks, visualize=False)

        print "Color Segmentation complete."

    def visulize_regions(self, objectSegmentation=None):
        """ 
        Visualizes regions recognized by the color segmentation.

        Keyword arguments:
        objectSegmentation -- reference to the ObjectSegmentation class which includes marker and meal regions.

        Returns:
        tuple with the following elements:
        - image with marked regions
        - list of all meal regions
        - list of meal areas
        """


        if self.originalImage is None:
            raise AttributeError("originalImage is not initialized. Run process_image() first")

        # Convert image to back to color to enable colored markers
        tempImage = np.copy(self.originalImage)        

        #mark food
        images = []     
        titles = []   
        for region, image in self.mealImages:
            color = (randint(0,255), randint(0,255), randint(0,255))
            cv.rectangle(tempImage,region.upperLeft,region.lowerRight,color,2)
            images.append(image)
            titles.append(self.__calculate_meal_size(region, objectSegmentation))

        return (tempImage, images, titles)

    def __calculate_meal_size(self, mealRegion, objectSegmentation):
        """
        Calculates the meal size based on the marker size.
        Keyword arguments:
        mealRegion -- meal region
        objectSegmentation -- reference to the ObjectSegmentation class which includes marker and meal regions.

        returns:
        size for meal region
        """

        if objectSegmentation is None:
            return "?"
        return objectSegmentation.calculate_meal_real_area(mealRegion=mealRegion)

    def __find_objects_of_masks(self, masks, visualize=False):
        """ Searches for objects in a list of image masks."""


        # contains all previous masks. Is initialized completely black.
        previousMasks = np.zeros_like(masks[0])

        maskNumber = 1
        objects = []

        for mask in masks:            
            # subtract the previous masks from this mask, so that we do not get any duplicate areas.
            mask = mask - previousMasks
            

            # are there any remaining parts
            if utils.is_image_black(mask):
                continue
            
            # add the new feature to previousMasks
            previousMasks = cv.bitwise_or(previousMasks, mask)

            # apply edge detection 
            mask = self.apply_edge_detection(mask, iterations=2, showSteps=False, imageConversion=cv.COLOR_GRAY2RGB)

            # search for objects in the mask
            utils.display_images([mask], None, titles=["Mask " + str(maskNumber)], rows=1, columns=1, show=visualize)
            for roi, roiImg, area in self.yield_image_objects(mask, areaThreshold=1000):
                utils.display_images([roiImg], titles=["Object Mask" + str(maskNumber) + "Area: " + str(area)], rows=1, columns=1, show=visualize)
                objects.append((roi, roiImg))

            maskNumber += 1

        return objects

    def __sort_image_masks(self, masks, visualize=False):
        """ Sorts and filters the image masks based on the area (mask sum) of the mask region."""

        # create a new list with the sum of the mask and the mask itself.
        # a higher sum means a mask with more white on it.
        sumsOfMasks = [(np.sum(mask), mask) for mask in masks]

        # eliminate (almost) completely black masks
        sumsOfMasks = [item for item in sumsOfMasks if not utils.is_image_black(None, imageSum=item[0])]

        # sort the sumOfMasks list by ordering the mask sum
        # so that masks with small white areas appear first.
        sumsOfMasks = sorted(sumsOfMasks, key=operator.itemgetter(0), reverse=False)

        

        # Show sorted masks tempList = [(AREAS), (MASKS)]
        titles, masks = ([t for t, _ in sumsOfMasks], [m for _, m in sumsOfMasks])
        utils.display_images(masks, None, titles, show=visualize) # since all masks are grayscale no conversion needed

        return masks

    def __calculate_mean_color_list(self):
        """ Deprecated (slow as hell)"""
        colors = []
        xCounter = yCounter = 0
        h, w, _ = self.originalImage.shape
        columns = int(w / ColorSegmentation.COLOR_WINDOW_SIZE[0]) #columns
        rows = int(h / ColorSegmentation.COLOR_WINDOW_SIZE[1])
        print "Size {0}x{1}".format(w, h) 
        for row in xrange(rows):
            print "Row:",row
            for column in xrange(columns):
                print "Column:",column
                # get the cropped part
                x = column * ColorSegmentation.COLOR_WINDOW_SIZE[0]
                y = row * ColorSegmentation.COLOR_WINDOW_SIZE[1]
                
                img = utils.crop_image(self.originalImage, x, y, ColorSegmentation.COLOR_WINDOW_SIZE[0], ColorSegmentation.COLOR_WINDOW_SIZE[1])
                #cv.imshow("{0}x{1}".format(w, h), img)
                #cv.waitKey(0)
                dominantColor = utils.get_modified_median_cut_dominant_color(img)
                colors.append(dominantColor)
                print "Dominant color for {0}x{1}: {2},{3},{4}".format(x,y, dominantColor[0], dominantColor[1], dominantColor[2])

                # for visualization print the color to the corresponding sector
                upperLeft = (x,y)
                lowerRight = (x + ColorSegmentation.COLOR_WINDOW_SIZE[0], y + ColorSegmentation.COLOR_WINDOW_SIZE[1])
                color = (dominantColor[2], dominantColor[1], dominantColor[0])
                cv.rectangle(self.colorMap, upperLeft, lowerRight, color, thickness=-1)

        print "Finished"
        return colors

    def __color_filter_image(self, colors, visualize=False):
        """ 
        Tries to find objects in the image using color filtering by applying the color-list.

        Keyword arguments:
        colors -- the colors to filter
        visualize -- show the filtering process

        returns: image masks
        """
        hsvImage = cv.cvtColor(self.originalImage, cv.COLOR_BGR2HSV)
        resultImage = None
        masks = []
        titles = []
        for color in colors:
            # convert color from RGB tuple to np HSV array
            color = np.uint8([[color]])
            hsvColor = cv.cvtColor(color, cv.COLOR_RGB2HSV)

            # define the thresholds for the color values
            hue = hsvColor[0][0][0]
            thLower = max(0, hue - 10)
            thUpper = min(179, hue + 10)
            thLowerColor = np.array([thLower, 100, 100])
            thUpperColor = np.array([thUpper, 255, 255])

            # threshold image with the color
            mask = cv.inRange(hsvImage, thLowerColor, thUpperColor)
            masks.append(mask)
            titles.append("Filter RGB: {0},{1},{2}".format(color[0][0][0], color[0][0][1], color[0][0][2]))
            if resultImage is None:
                resultImage = mask
            else:
                resultImage = cv.bitwise_or(resultImage, mask)
            # use the mask to bring out only the elements with the specified color            
            #cv.imshow('mask',mask)
            #cv.imshow('res',resultImage)
            cv.imwrite("C:/Users/felix/Desktop/Segmentation/results/resultsMask_{0}{1}{2}.jpg".format(color[0][0][0], color[0][0][1], color[0][0][2]), mask)
            cv.imwrite("C:/Users/felix/Desktop/Segmentation/results/resultsMaskResult_{0}{1}{2}.jpg".format(color[0][0][0], color[0][0][1], color[0][0][2]), mask)

        if visualize:
            # apply masks
            images = [cv.cvtColor(cv.bitwise_and(self.originalImage, self.originalImage, mask=m), cv.COLOR_BGR2RGB) for m in masks]

            # append filtered image
            images.append(cv.cvtColor(cv.bitwise_and(self.originalImage, self.originalImage, mask=resultImage), cv.COLOR_BGR2RGB))
            titles.append("combined images")
            images.extend(masks)
            titles.extend(["Image Mask " + str(i) for i in xrange(len(masks))])
            utils.display_images(images, None, titles)
        return masks