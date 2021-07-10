from enum import Enum

from .object import Classes

class Config:
    """Base configuration. Inherit to class to access all predefined variables/functions from the Config class.
    
    Add additional variables to overwrite or extend the class


    """
    training_img_dir = "../data/FlyingObjectDataset_10K/training"
    validation_img_dir = "../data/FlyingObjectDataset_10K/validation"
    testing_img_dir = "../data/FlyingObjectDataset_10K/testing"
    spiral_path = "../data/lab2/spiral.dat"
    
    SEED = 420
    GPU = -1                 # GPU ID or set as -1 in case of using any GPU
    IMAGE_WIDTH = 128       # image width
    IMAGE_HEIGHT = 128      # image height
    IMAGE_CHANNEL = 3       # image channel
    NUM_WORKERS = 4
    BATCH_SIZE = 32         # batch size
    SAVE_EVERY = 1         # save after each epoch
    TENSORBORD_DIR = "logs/"
    
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
    

    def todict(self):
        """Convert variables in class object to dictionary"""
        data = list(filter(lambda x: True if "__" not in x and not callable(getattr(self, x)) else False, self.__dir__()))
        
        return {x:getattr(self,x) for x in data}
    
    def __repr__(self):
        """Print variables in class"""
        #return ",\n".join([''.join([x,getattr(self,x)]) for x in self.__dir__() if "__" not in x and not callable(getattr(self, x))])
        data = list(filter(lambda x: True if "__" not in x and not callable(getattr(self, x)) else False, self.__dir__()))
        
        return str({x:getattr(self,x) for x in data})