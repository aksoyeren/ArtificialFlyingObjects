from utils.globals import Config

class LabConfig(Config):
    """Configuration for lab. Inherit variables from the Config class.
    
    Add additional variables to overwrite or extend the class


    """
    spiral_path = "../data/lab2/spiral.dat"
    ae_train = "../data/lab2/ae.train"
    ae_test = "../data/lab2/ae.test"
    CLASSES = 6
    AE_CLASSES = [1,2,3,4,5,6,7,8,9]
    AE_NUM_CLASSES = len(AE_CLASSES)