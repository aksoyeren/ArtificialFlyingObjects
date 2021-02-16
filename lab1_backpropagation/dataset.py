import h5py
from pathlib import Path
import torch
from torch.utils.data import Dataset,DataLoader
from easydict import EasyDict as edict
import numpy as np
import os
import re
import imageio
from skimage.transform import resize
import torchvision

class H5DataLoader:
    "Custom loader where we can init dataset for training, validation and testing"
    def __init__(self, **hparams):
        self.hparams = edict(hparams)
        
        
    def train_dataloader(self):
        ds = HDF5Dataset(self.hparams.dataset, step='training')
        
        dataloader = DataLoader(ds,
                                batch_size=self.hparams.BATCH_SIZE,
                                shuffle=True,
                                num_workers=self.hparams.NUM_WORKERS)
        return dataloader
    
    def validation_dataloader(self):
        ds = HDF5Dataset(self.hparams.dataset, step='validation')
        
        dataloader = DataLoader(ds,
                                batch_sizeself.hparams.BATCH_SIZE,
                                shuffle=False,
                                num_workers=self.hparams.NUM_WORKERS)
        return dataloader
    
    def test_dataloader(self):
        ds = HDF5Dataset(self.hparams.dataset, step='testing')
        
        dataloader = DataLoader(ds,
                                batch_size=self.hparams.BATCH_SIZE,
                                shuffle=False,
                                num_workers=self.hparams.NUM_WORKERS)
        return dataloader

class HDF5Dataset(Dataset):
    def __init__(self, file_path, step='training',transform=None):
        super().__init__()
        self.transform = transform
    
        p = Path(file_path)
        assertp.is_file(), "The path is not a valid h5 file!"
        df = h5py.File(p, 'r')

        assert step in list(df.keys()), "Could not find step in keys!"
        self.data = df[step]['data']
        self.labels = df[step]['label']
        
    def __getitem__(self, index):
        # get data
        x = self.data[index].T # Transpose to pytorch input
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)
        x = x.float()
        # get label
        y = self.labels[index]
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return self.data.shape[0]
    
    def close(self):
        self.df.close()

class ImageDataLoader:
    "Custom loader where we can init dataset for training, validation and testing"
    def __init__(self, **hparams):
        self.hparams = edict(hparams)
        self.img_shape = (self.hparams.IMAGE_HEIGHT,self.hparams.IMAGE_WIDTH,self.hparams.IMAGE_CHANNEL)
        
    def train_dataloader(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda img:resize(img, self.img_shape)),
            torchvision.transforms.ToTensor()
        ])
        
        ds = ImageDataset(self.load_image_paths(self.hparams.training_img_dir), transform=transform)
        
  
        dataloader = DataLoader(ds,
                                batch_size=self.hparams.BATCH_SIZE,
                                shuffle=False,
                                num_workers=self.hparams.NUM_WORKERS)
        return dataloader
    
    def validation_dataloader(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda img:resize(img, self.img_shape)),
            torchvision.transforms.ToTensor()
        ])
        
        ds = ImageDataset(self.load_image_paths(self.hparams.validation_img_dir), transform=transform)
        
        dataloader = DataLoader(ds,
                                batch_size=self.hparams.BATCH_SIZE,
                                shuffle=False,
                                num_workers=self.hparams.NUM_WORKERS)
        return dataloader
    
    def test_dataloader(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda img:resize(img, self.img_shape)),
            torchvision.transforms.ToTensor()
        ])
        ds = ImageDataset(self.load_image_paths(self.hparams.testing_img_dir), transform=transform)
        
        dataloader = DataLoader(ds,
                                batch_size=self.hparams.BATCH_SIZE,
                                shuffle=False,
                                num_workers=self.hparams.NUM_WORKERS)
        return dataloader
    
    def load_image_paths(self,image_path):
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
    
class ImageDataset(Dataset):
    def __init__(self, data,transform=None):
        super().__init__()
        self.transform = transform
        self.data = [path + '/' + filename for path,filename,*_ in data]
        self.labels = [label for _,_,_,_,label,_ in data]    #name.split('_',1)[0]
        
    def __getitem__(self, index):
        # get data
        x = imageio.imread(self.data[index])
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)
        x = x.float()
   
        # get label
        y = self.labels[index]
   
        y = torch.tensor(int(y))
        return (x, y)

    def __len__(self):
        return len(self.data)