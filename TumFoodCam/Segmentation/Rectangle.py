class Rectangle(object):
    """ Rectangle wrapper.
    """

    def __init__(self, upperLeftPoint, size):
        self.width, self.height = size
        self.upperLeft = upperLeftPoint
        self.lowerRight = (upperLeftPoint[0] + self.width, upperLeftPoint[1] + self.height)

    def contains(self, other):
        # TODO
        pass