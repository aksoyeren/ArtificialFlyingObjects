import numpy as np
from easydict import EasyDict as edict

def flying_objects_config():

    cfg = edict()

    cfg.dataset = "../data/flyingData_lab1.hdf5"

    #cfg.NUM_CLASS = len(cfg.CLASSES)    # number of classes
    cfg.GPU = 0                 # GPU ID or set as -1 in case of using CPU
    cfg.NUM_CLASS=12
    cfg.DROPOUT_PROB = 0.5      # Probability to keep a node in dropout
    cfg.IMAGE_WIDTH = 128       # image width
    cfg.IMAGE_HEIGHT = 128      # image height
    cfg.IMAGE_CHANNEL = 3       # image channel
    cfg.NUM_EPOCHS = 10        # epoch number
    cfg.NUM_WORKERS = 4
    cfg.BATCH_SIZE = 32         # batch size
    cfg.LEARNING_RATE = 0.001  # learning rate
    cfg.LR_DECAY_FACTOR = 0.1   # multiply the learning rate by this factor
    cfg.PRINT_EVERY = 20        # print in every 50 epochs
    cfg.SAVE_EVERY = 1         # save after each epoch
    cfg.DEBUG_MODE = True      # print log to console in debug mode
    cfg.DATA_AUGMENTATION = True   # Whether to do data augmentation

    return cfg
