from glob import glob
from torch.utils.data import Dataset
from skimage.transform import resize
from torchvision.transforms import functional as ff
import os
from glob import glob
from PIL import Image
import numpy as np
import torch
import random

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
    
class SegmentationDataset(Dataset):
    def __init__(self, data_dir,img_shape=None,transform=None, predict=False, shuffle=True):
        super().__init__()
        self.transform = transform
        self.img_shape = img_shape
        self.predict = predict
        self.images = list(sorted(glob(os.path.join(data_dir, 'image', '*.png'))))
            
        self.labels = list(sorted(glob(os.path.join(data_dir, 'gt_image', 'gt_*.png'))))

        # Shuffle dataset
        if shuffle:
            c = list(zip(self.images, self.labels))
            random.Random(42).shuffle(c)
            self.images, self.labels = zip(*c)
        
        if not self.predict: 
            assert len(self.images) == len(self.labels), "Numbers of image and ground truth labels are not the same>> image nbr: %d gt nbr: %d"  % (n_image, n_labels)
        
        
    def __getitem__(self, index):
       
        # Load image path
        image_file = self.images[index]
        
        # Preprocessing
        image = Image.open(image_file)
        image = image.resize(self.img_shape)
        image = np.asarray(image)/255#).astype('float32') 
        
        bkgnd_image =  np.ones(self.img_shape)
        bkgnd_image  = bkgnd_image - image[:,:,0]
        bkgnd_image  = bkgnd_image - image[:,:,1]
        bkgnd_image  = bkgnd_image - image[:,:,2]
        image = np.dstack((image,bkgnd_image))

        if not self.predict: 
            gt_image_file = self.labels[index]
            
            # read labels
            gt_image = Image.open(gt_image_file)
            gt_image = gt_image.resize(self.img_shape)
            gt_image = np.asarray(gt_image)/255

            #create background image
            bkgnd_image =  np.ones(self.img_shape)
            bkgnd_image  = bkgnd_image - gt_image[:,:,0]
            bkgnd_image  = bkgnd_image - gt_image[:,:,1]
            bkgnd_image  = bkgnd_image - gt_image[:,:,2]
            gt_image = np.dstack((gt_image,bkgnd_image))

            labels = torch.argmax(ff.to_tensor(gt_image).float(), dim=0)
            return ff.to_tensor(image).float(), labels
        
        return ff.to_tensor(image).float()
                
    def __len__(self):
        return len(self.images)
    
    

class ClassificationDataset(Dataset):
    def __init__(self, cfg, classes, transform=None, fineGrained=False):
        super().__init__()
        self.transform = transform
        self.hparams = cfg
        self.classes = classes
        self.fineGrained = fineGrained
        self.images = images = glob(os.path.join(data_folder, 'image', '*.png'))
        
    def __getitem__(self, index):
       
        # Load image path
        image_file = self.images[index]
  
        image = Image.open(image_file)
        image = image.resize((self.hparams.IMAGE_HEIGHT, self.hparams.IMAGE_WIDTH))
        image = np.asarray(image)/255

        # read labels from image_file names
        labels = np.zeros(shape=(len(self.classes)), dtype=np.float32)
        path, img_name = os.path.split(image_file)
        names = img_name.split(".")[0].split("_")

        currLabel = names[1] + "_" + names[2] if self.fineGrained else names[1]
        
        if np.isin(currLabel, self.classes):
            loc = self.classes.index(currLabel)
            labels[loc] = 1
        else:
            raise ValueError("ERROR: Label " + str(currLabel) + " is not defined!")

        return image, labels
                
    def __len__(self):
        return len(self.images)