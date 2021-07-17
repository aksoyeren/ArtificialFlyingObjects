from DL_labs.utils.globals import Config


class LabConfig(Config):
    """ """
    COCO_DATASET = "/var/metrics/DL_DATASET/coco"
    
    training_img_dir = f"{COCO_DATASET}/train2017"
    validation_img_dir = f"{COCO_DATASET}/validation2017"
    training_annotations = f"{COCO_DATASET}/annotations/instances_train2017.json"
    
    