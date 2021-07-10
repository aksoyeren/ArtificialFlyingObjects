from DL_labs.utils.globals import Config


class LabConfig(Config):
    """Configuration for lab. Inherit variables from the Config class.
    
    Add additional variables to overwrite or extend the class


    """
    training_img_dir = "../data/FlyingObjectDataset_10K/training"
    validation_img_dir = "../data/FlyingObjectDataset_10K/validation"
    testing_img_dir = "../data/FlyingObjectDataset_10K/testing"
