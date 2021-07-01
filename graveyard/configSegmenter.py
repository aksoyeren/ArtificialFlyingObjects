import numpy as np
from easydict import EasyDict as edict

def flying_objects_config():

  cfg = edict()

  cfg.training_data_dir = "../data/FlyingObjectDataset_10K/training"
  cfg.validation_data_dir = "../data/FlyingObjectDataset_10K/validation"
  cfg.testing_data_dir = "../data/FlyingObjectDataset_10K/testing"
  cfg.fineGrained = False

  if cfg.fineGrained:
      # classes
      cfg.CLASSES = [
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
          'circular_yellow']
  else:
      # classes
      cfg.CLASSES = [
          'square',
          'triangle',
          'circular',
          'background']

  cfg.NUM_CLASS = len(cfg.CLASSES)    # number of classes

  cfg.GPU = 1                 # GPU ID
  cfg.DROPOUT_PROB = 0.5      # Probability to keep a node in dropout
  cfg.IMAGE_WIDTH = 128       # image width
  cfg.IMAGE_HEIGHT = 128      # image height
  cfg.IMAGE_CHANNEL = 3       # image channel
  cfg.NUM_EPOCHS = 1        # epoch number
  cfg.BATCH_SIZE = 32         # batch size
  cfg.LEARNING_RATE = 0.001  # learning rate
  cfg.LR_DECAY_FACTOR = 0.1   # multiply the learning rate by this factor
  cfg.PRINT_EVERY = 20        # print in every 50 epochs
  cfg.SAVE_EVERY = 1         # save after each epoch
  cfg.DEBUG_MODE = True      # print log to console in debug mode
  cfg.DATA_AUGMENTATION = True   # Whether to do data augmentation

  return cfg
