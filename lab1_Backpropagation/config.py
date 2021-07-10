from DL_labs.utils.globals import Config

class LabConfig(Config):
    """Configuration for lab. Inherit variables from the Config class.
    
    Add additional variables to overwrite or extend the class


    """
    NUM_CLASS = 10    # number of classes
    IMAGE_WIDTH = 28       # image width
    IMAGE_HEIGHT = 28      # image height
    IMAGE_CHANNEL = 3       # image channel