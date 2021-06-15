import numpy as np
from easydict import EasyDict as edict
class classificationComplexity():
    
    def __init__(self,fineGrained=False):
        self.fineGrained = fineGrained
    
    @property
    def fineGrained(self):
        return [
            'square_red',
            'square_green',
            'square_blue',
            'square_yellow',
            'triangle_red',
            'triangle_green',
            'triangle_blue',
            'triangle_yellow',
            'circular_red',
            'circular_green',
            'circular_blue',
            'circular_yellow'
        ] if self._fineGrained else [
            'square',
            'triangle',
            'circular',
            'background'
        ]
    @fineGrained.setter
    def fineGrained(self, fineGrained):
        self._fineGrained = fineGrained
        
        return self.fineGrained
        
    def __str__(self):
        return self._fineGrained
    
    def __len__(self):
        return len(self.fineGrained)
        
def flying_objects_config():

    cfg = edict()
    cfg.fineGrained = classificationComplexity(False).fineGrained
    #cfg.CLASSES = classificationComplexity(cfg.fineGrained)
    
    cfg.training_img_dir = "../data/FlyingObjectDataset_10K/training"
    cfg.validation_img_dir = "../data/FlyingObjectDataset_10K/validation"
    cfg.testing_img_dir = "../data/FlyingObjectDataset_10K/testing"
    
    cfg.NUM_CLASS = len(cfg.fineGrained)    # number of classes
    cfg.GPU = -1                 # GPU ID or set as -1 in case of using any GPU
    cfg.IMAGE_WIDTH = 128       # image width
    cfg.IMAGE_HEIGHT = 128      # image height
    cfg.IMAGE_CHANNEL = 3       # image channel
    cfg.NUM_WORKERS = 4
    cfg.BATCH_SIZE = 32         # batch size
    cfg.SAVE_EVERY = 1         # save after each epoch
    cfg.TENSORBORD_DIR = "logs/"
    #cfg.DATA_AUGMENTATION = True   # Whether to do data augmentation

    return cfg
