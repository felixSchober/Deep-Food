class Classifier(object):
    """Base class for all classifiers."""

    def __init__(self, imageSize, name, description, testData, grayscale=True, transform=None):
        """
        Keyword arguments: 
        imageSize -- the desired image size to be loaded.
        name -- name of the classifier
        description -- description of the classifier
        testData -- reference to the test data class
        grayscale -- load images in grayscale
        transform -- custom method that transforms every image during loading
        """


        self.imageSize = imageSize
        self.transform = transform
        self.name = name
        self.description = description
        self.testData = testData
        self.trained = False
        self.modelSaver = None
        self.tester = None
        self.grayscale = grayscale

    def train_and_evaluate(self, save=True):
        raise NotImplementedError("abstract class")

    def predict(self, images):
        raise NotImplementedError("abstract class")
    
    def save(self):  
        raise NotImplementedError("abstract class")




