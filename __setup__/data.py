import h5py
from glob import glob 
from skimage.transform import resize
import imageio
import os
import numpy as np
from typing import List, Tuple, TypeVar

T = TypeVar('T')  # Can be anything
S = TypeVar('S', str, bytes)  
F = TypeVar('F', int, float) 

class HDF5File:
    def __init__(self, img_shape:Tuple[F, F, F], classes:List[T], db_path='../data/flyingData_lab1.hdf5'):
        self.db_path = db_path
        self.img_shape = img_shape
        self.img_classes = classes
    
    def create_datasets(self, dir_paths:List[S]):
        for dir_path in dir_paths:
            self.create_dataset(dir_path)
        
    def create_dataset(self, dir_path:S):
        group_name = dir_path.rsplit("/",1)[1]
        print("Processing:", group_name)
        
        imgs = self.get_img_paths(dir_path)
        with h5py.File(self.db_path, 'a') as f:
            grp = f.create_group(group_name)
            self.create_group(grp, imgs)
            self.images_to_group(grp, imgs)
        
        print("Finished loading data..")
    def create_group(self, f, imgs:List[T]):
        
        num_imgs = len(imgs)
        data_shape = (num_imgs, *self.img_shape )
        label_shape = (num_imgs, len(self.img_classes))
        
        #grp = f.create_group(group_name)
        f.create_dataset('data', data_shape, np.float)
        f.create_dataset('label', label_shape, np.float)

    def images_to_group(self,group, imgs):
        for i, img_path in enumerate(imgs):
            print(f"Processing {i}", end="\r")
            image = resize(imageio.imread(img_path), self.img_shape)
            
            labels = np.zeros(shape=(len(self.img_classes)), dtype=np.float32)
                       
            img_path, img_name = os.path.split(img_path)
            fn, ext = img_name.split(".")
            names = fn.split("_")
                       
            currLabel = names[1] + "_" + names[2]
            
            assert np.isin(currLabel, self.img_classes), f"ERROR: Label {currLabel} is not defined!"
            loc = self.img_classes.index(currLabel)
            labels[loc] = 1
        
            group["data"][i, ...] = image
            group["label"][i, ...] = labels
    
    def get_img_paths(self, dir_path:S):
        return glob(os.path.join(dir_path, 'image', '*.png'))
    
    def get_groups(self):
        with h5py.File(self.db_path, 'r') as f:
            return list(f.keys())
    
        