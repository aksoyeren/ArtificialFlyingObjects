import numpy as np
from easydict import EasyDict as edict

def flying_objects_config():
    """ """

  cfg = edict()

  cfg.training_data_dir = "../data/FlyingObjectDataset_10K/training"
  cfg.validation_data_dir = "../data/FlyingObjectDataset_10K/validation"
  cfg.testing_data_dir = "../data/FlyingObjectDataset_10K/testing"
    
  cfg.GPU = 0                 # GPU ID
  cfg.DROPOUT_PROB = 0.5      # Probability to keep a node in dropout
  cfg.IMAGE_WIDTH = 64       # image width
  cfg.IMAGE_HEIGHT = 64      # image height
  cfg.IMAGE_CHANNEL = 3       # image channel
  cfg.NUM_EPOCHS = 5        # epoch number
  cfg.BATCH_SIZE = 30         # batch size
  cfg.SEQUENCE_LENGTH = 10         # length of each sequence
  cfg.LEARNING_RATE = 0.01  # learning rate
  cfg.LR_DECAY_FACTOR = 0.1   # multiply the learning rate by this factor
  cfg.PRINT_EVERY = 20        # print in every 50 epochs
  cfg.SAVE_EVERY = 1         # save after each epoch
  cfg.DEBUG_MODE = True      # print log to console in debug mode
  cfg.DATA_AUGMENTATION = True   # Whether to do data augmentation

  return cfg
