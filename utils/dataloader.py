#import h5py
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import numpy as np
import os
import re
import imageio

import torchvision
class H5DataLoader(DataLoader):
    """ """
    "Custom loader where we can init dataset for training, validation and testing"
    def __init__(self, **hparams):
        self.hparams = edict(hparams)
        
        
    def train_dataloader(self):
        """ """
        ds = HDF5Dataset(self.hparams.dataset, step='training')
        
        dataloader = DataLoader(ds,
                                batch_size=self.hparams.BATCH_SIZE,
                                shuffle=True,
                                num_workers=self.hparams.NUM_WORKERS)
        return dataloader
    
    def validation_dataloader(self):
        """ """
        ds = HDF5Dataset(self.hparams.dataset, step='validation')
        
        dataloader = DataLoader(ds,
                                batch_sizeself.hparams.BATCH_SIZE,
                                shuffle=False,
                                num_workers=self.hparams.NUM_WORKERS)
        return dataloader
    
    def test_dataloader(self):
        """ """
        ds = HDF5Dataset(self.hparams.dataset, step='testing')
        
        dataloader = DataLoader(ds,
                                batch_size=self.hparams.BATCH_SIZE,
                                shuffle=False,
                                num_workers=self.hparams.NUM_WORKERS)
        return dataloader


class ImageDataLoader:
    """ """
    "Custom loader where we can init dataset for training, validation and testing"
    def __init__(self, config, datasetModel=None,**hparams):
        
        assert datasetModel != None, "Input a valid dataset class to use!"
        self.hparams = edict(config)
        self.img_shape = (self.hparams.IMAGE_HEIGHT,self.hparams.IMAGE_WIDTH,self.hparams.IMAGE_CHANNEL)
        
        self.datasetModel = datasetModel
        
        self.training_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda img:resize(img, self.img_shape)),
            torchvision.transforms.ToTensor()
        ])
        
        
        self.test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda img:resize(img, self.img_shape)),
            torchvision.transforms.ToTensor()
        ])
        
    def train_dataloader(self):
        """ """
        ds = self.datasetModel(self.hparams, transform=self.training_transforms)
        
        dataloader = DataLoader(ds,
                                batch_size=self.hparams.BATCH_SIZE,
                                shuffle=False,
                                num_workers=self.hparams.NUM_WORKERS)
        return dataloader
    
    def validation_dataloader(self):
        """ """
        ds = self.datasetModel(self.hparams, transform=self.validation_transforms)
        
        dataloader = DataLoader(ds,
                                batch_size=self.hparams.BATCH_SIZE,
                                shuffle=False,
                                num_workers=self.hparams.NUM_WORKERS)
        return dataloader
    
    def test_dataloader(self):
        """ """
    
        ds = self.datasetModel(self.hparams, transform=self.test_transforms)
        
        dataloader = DataLoader(ds,
                                batch_size=self.hparams.BATCH_SIZE,
                                shuffle=False,
                                num_workers=self.hparams.NUM_WORKERS)
        return dataloader
    
    def load_image_paths(self,image_path:str) -> "np.ndarray":
        """Load dataloader images.

        :param image_path: str:
        :param image_path: str:
        :param image_path: str:
        :param image_path: str:
        :param image_path: str:
        :param image_path: str:
        :param image_path: str:
        :param image_path:str: 

        """
        if self.hparams.fineGrained:
            filenameContext = re.compile('^(\w+?)\_{1}(\d{6})\_{1}([\w]+?)\_{1}(\d{6})',re.DOTALL) 
            img_fix_category = lambda img: [self.hparams.CLASSES.index(img[2]) if i == 2 else img[i] for i, item in enumerate(img)]
        else:
            filenameContext = re.compile('^(\w+?)\_{1}(\d{6})\_{1}([\w]+?)\_{1}([\w]+?)\_{1}(\d{6})',re.DOTALL)
            img_fix_category = lambda img: [self.hparams.CLASSES.index(img[2]) if i == 2 else img[i] for i, item in enumerate(img)]
        
      
        img_files_collect = lambda path: [
            (
                dp,f, *img_fix_category(context)
            ) 
            for dp, dn, filenames in os.walk(path) 
            for f in filenames 
            if os.path.splitext(f)[1] == '.png' and (context := filenameContext.split(f)[1:5])
        ]
        imgs = img_files_collect(image_path)
        assert len(imgs) > 0, f"No images loaded from {image_path} path!"
        imgs = sorted(imgs, key=lambda x:x[1])
        return np.array(imgs)
    

    