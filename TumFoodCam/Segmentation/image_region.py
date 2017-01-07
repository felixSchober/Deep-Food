import cv2 as cv
import numpy as np
from segmentation import Rectangle as r
import misc.utils as utils

class ImageRegion(object):
    """Class to represent an image region."""
    
    def __init__(self, contour=None, upperLeft=None, lowerRight=None, cutmask=None):
        """ Creates an image Region object.
        Region can be constructed by contour, combination of upperLeft and lowerRight or cutmask rectangle.

        Keyword arguments:
        contour -- OpenCV contour
        upperLeft -- Tuple of upperLeft Point coordinates
        lowerRight -- Tuple of lowerRight Point coordinates
        cutmask -- rectangle of region
        """
        
        if contour is None and cutmask is None and (upperLeft is None or lowerRight is None):
            raise AttributeError("Either provide upperLeft and lowerRight or a cut mask")

        self.contour = contour
        
        if cutmask is None and upperLeft is not None and lowerRight is not None:
            self.upperLeft = upperLeft
            self.lowerRight = lowerRight 
            width = lowerRight[0] - upperLeft[0]
            height = lowerRight[1] - upperLeft[1]
            self.cutmask = r.Rectangle(upperLeft, (width, height))
        elif cutmask is not None:
            self.cutmask = cutmask
            self.upperLeft = cutmask.upperLeft
            self.lowerRight = cutmask.lowerRight
        else:
            x,y,w,h = cv.boundingRect(contour)      
            self.cutmask = r.Rectangle((x, y), (w, h))
            self.upperLeft = self.cutmask.upperLeft
            self.lowerRight = self.cutmask.lowerRight
        

    def create_mask(self, shape):
        """ Creates a mask for the image with the image region."""

        # make sure shape is only 1 channel
        shape = (shape[0], shape[1])

        # create a new black image
        mask = np.zeros(shape, dtype=np.uint8)

        # draw the contour as a white filled area
        cv.drawContours(mask, [self.contour], 0, (255,255,255), thickness=-1)
        return mask

    def crop_image_region(self, image):
        """ Crops an image to the given imageRegion and returns it.
        """
        x, y = self.upperLeft
        w, h = self.get_dimension()
        return utils.crop_image(image, x, y, w, h)
    
    def get_roi_image(self, image):
        """ Returns the image with only the image region visible."""

        m = self.create_mask(image.shape)
        # Apply the mask
        return cv.bitwise_and(image, image, mask=m)

    def overlay_rectangle(self, image, alpha=0.1, color=(0,255,0)):
        """ 
        Overlays a transparent rectangle.

        Keyword arguments:
        image -- image to overlay the rectangle on
        alpha -- alpha value of the rectangle
        color -- BGR color of the rectangle
        """

        result = image.copy()
        overlay = image.copy()

        cv.rectangle(overlay, self.upperLeft, self.lowerRight, color, -1)
        
        cv.addWeighted(overlay, alpha, result, 1-alpha, 0, result)
        return result
        
    def is_similar_to(self, other):
        """Deprecated (not implemented)"""

        if other is None:
            return False

        c1 = self.cutmask
        c2 = other.cutmask
        comparedMeasures = [(c1.width, c2.width), (c1.height, c2.height), (c1.upperLeft[0], c2.upperLeft[0]), (c1.upperLeft[1], c2.upperLeft[1])]

        for i in range(len(comparedMeasures)/2):
            if abs(comparedMeasures[i][0] - comparedMeasures[i][1]) > 2:
                return False
        return True

    def contains_image_region(self, regions, newRegion):
        """Deprecated (not implemented)"""
        for region in regions:
            c1 = region.cutmask
            c2 = newRegion.cutmask

            # check if similar
            if region.is_similar_to(newRegion):
                return True

            # check if contained by existing region
            if c1.contains(c2):
                return True
        return False
    
    def get_ratio(self):
        """ Get the aspect ratio of height and width."""
        return self.cutmask.height / self.cutmask.width

    def get_dimension(self):
        """ Get the dimension of the image region."""
        return (self.cutmask.width, self.cutmask.height)