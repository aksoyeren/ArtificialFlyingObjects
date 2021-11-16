from utils.globals import Config
from utils.object import Classes

class LabConfig(Config):
    """Configuration for lab. Inherit variables from the Config class.
    
    Add additional variables to overwrite or extend the class


    """
    training_img_dir = "../data/FlyingObjectDataset_10K/training"
    validation_img_dir = "../data/FlyingObjectDataset_10K/validation"
    testing_img_dir = "../data/FlyingObjectDataset_10K/testing"

    def __init__(self):
        self.classification(False)
        
    def classification(self, fineGrained):
        """Load classification classes with finegrained or not

        :param fineGrained: 

        """
        CLASSIFICATION = Classes(fineGrained)
        self.CLASSES = CLASSIFICATION.classes
        self.fineGrained = CLASSIFICATION.fineGrained
        self.NUM_CLASSES = len(self.CLASSES) 