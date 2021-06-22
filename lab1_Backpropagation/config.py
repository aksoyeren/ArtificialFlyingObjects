import numpy as np
from easydict import EasyDict as edict

def flying_objects_config():

    cfg = edict()
    cfg.NUM_CLASS = 10    # number of classes
    cfg.GPU = -1                 # GPU ID or set as -1 in case of using any GPU
    cfg.IMAGE_WIDTH = 64       # image width
    cfg.IMAGE_HEIGHT = 64      # image height
    cfg.IMAGE_CHANNEL = 3       # image channel
    cfg.NUM_WORKERS = 4
    cfg.BATCH_SIZE = 32         # batch size
    cfg.SAVE_EVERY = 1         # save after each epoch
    cfg.TENSORBORD_DIR = "logs/"
    return cfg
