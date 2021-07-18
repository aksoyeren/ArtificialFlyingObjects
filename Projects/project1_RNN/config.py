from DL_labs.utils.globals import Config


class LabConfig(Config):
    """ """
    training_img_dir = "../../data/FlyingObjectDataset_10K/training/image"
    validation_img_dir = "../../data/FlyingObjectDataset_10K/validation/image"
    testing_img_dir = "../../data/FlyingObjectDataset_10K/testing/image"
    
    SEQUENCE_LENGTH = 7