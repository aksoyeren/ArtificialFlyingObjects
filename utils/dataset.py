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

from . import utils

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
        self.labels = [label for _,_,_,_,label,_ in data]    
        
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
    def __init__(self, data_dir,img_shape=None,transform=None, predict=False, shuffle=True, augmentation=None):
        super().__init__()
        self.transform = transform
        self.augmentation = augmentation
        self.img_shape = img_shape
        self.predict = predict
        self.images = list(sorted(glob(os.path.join(data_dir, 'image', '*.png'))))
        self.labels = list(sorted(glob(os.path.join(data_dir, 'gt_image', 'gt_*.png'))))
        
        assert self.transform != None, "transform cant be empty!"
        
        # Shuffle dataset
        if shuffle:
            c = list(zip(self.images, self.labels))
            random.Random(42).shuffle(c)
            self.images, self.labels = zip(*c)
        
        if not self.predict: 
            assert len(self.images) == len(self.labels), "Numbers of image and ground truth labels are not the same>> image nbr: %d gt nbr: %d"  % (n_image, n_labels)
        
        
    def __getitem__(self, index): # TODO: How do I transform two images at once??
        # Note: Expect (B,H,W) for mask and (B,C,H,W) for image where C is the same size as number of classes
        # Load image path
        image_file = self.images[index]
        
        # Preprocessing
        image = utils.normalize(np.array(Image.open(image_file)))
        image = self.__segment_background(image)
        image = self.transform(image)

        if not self.predict: 
            gt_image_file = self.labels[index]
            
            # read labels
            gt_image = self.__segment_background(utils.normalize(np.array(Image.open(gt_image_file))))
            gt_image = self.transform(gt_image)
            labels = torch.argmax(gt_image, dim=0)
            
            return image.float(), labels
        return image.float()
                
    def __len__(self):
        return len(self.images)
    
    def __segment_background(self, img):
        bkgnd_image =  np.ones(self.img_shape)
        bkgnd_image  = bkgnd_image - img[:,:,0]
        bkgnd_image  = bkgnd_image - img[:,:,1]
        bkgnd_image  = bkgnd_image - img[:,:,2]
   
        img = np.dstack((img,utils.normalize(bkgnd_image)))
        return img

class ClassificationDataset(Dataset):
    def __init__(self, data_dir, classes,img_shape=None, transform=None, fineGrained=False, predict=False, shuffle=True):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.img_shape = img_shape
        self.classes = classes
        self.fineGrained = fineGrained
        self.images = images = glob(os.path.join(data_dir, 'image', '*.png'))
        
        assert self.transform != None, "transform cant be empty!"
        
    def __getitem__(self, index):
       
        # Load image path
        image_file = self.images[index]

        image = self.transform(Image.open(image_file))

        # read labels from image_file names
        labels = self.__extract_label(image_file)

        return image, labels
    
    def __len__(self):
        return len(self.images)
    
    def item(self,index):
        return self.__getitem__(index)
    
    def __extract_label(self, image_file):
        #labels = np.zeros(shape=(len(self.classes)), dtype=np.float32)
        path, img_name = os.path.split(image_file)
        names = img_name.split(".")[0].split("_")

        currLabel = names[1] + "_" + names[2] if self.fineGrained else names[1]
        
        if currLabel in self.classes:
            label = self.classes.index(currLabel)
        else:
            raise ValueError("ERROR: Label " + str(currLabel) + " is not defined!")
        
        return label
    
    
    
    
    

class LastFramePredictorDataset(Dataset): #(data_folder, image_shape, batch_size):
    def __init__(self, data_folder,img_shape=None, transform=None, fineGrained=False, predict=False, shuffle=True):
        super().__init__()
        self.data_folder = data_folder
        self.transform = transform
        self.img_shape = img_shape
        self.images, self.lastframe = self.image_sequence(sorted(glob(os.path.join(data_folder, 'image', '*.png'))))
        
        assert self.transform != None, "transform cant be empty!"
        
    def __getitem__(self, index):

        image_file = os.path.join(self.data_folder, "image", self.images[index])
        last_image = os.path.join(self.data_folder,  "image", self.lastframe[index]) 
        
        image = utils.normalize(np.array(Image.open(image_file)))
        image = self.transform(image)

        gt_image = utils.normalize(np.array(Image.open(last_image)))
        gt_image = self.transform(gt_image)
    
       
        yield image, gt_image
    
    def image_sequence(self, images):
        sequence = []
        lastframe = []

        last_frame_id = None
        last_image_name = None
        last_action_id = None
        for image in images:
            _, img_name = os.path.split(image)
            action_id, class_id, color_id, frame_id  = img_name.split(".")[0].split("_")

            sequence.append(img_name)
            if not action_id == last_action_id and last_action_id != None:
                lastframe.extend([last_image_name]*int(last_frame_id)) #*int(last_frame_id)
                last_action_id = action_id

            last_action_id = action_id
            last_frame_id = frame_id
            last_image_name = img_name

        return sequence, lastframe

class FutureFramePredictorDataset(Dataset): #(data_folder, image_shape, batch_size):
    
    def __init__(self, data_folder, sequence_length, img_shape=None, transform=None, fineGrained=False, predict=False, shuffle=True):
        super().__init__()
        self.data_folder = data_folder
        self.sequence_length = sequence_length
        self.transform = transform
        self.img_shape = img_shape
        self.images = self.image_sequence(sorted(glob(os.path.join(data_folder, 'image', '*.png'))))
        
        assert self.transform != None, "transform cant be empty!"
        
    def __getitem__(self, index):
        
        def _transform_time(data):
            new_data = None
            for image_file in data:
                image = utils.normalize(np.array(Image.open(os.path.join(self.data_folder, "image", image_file))))
                new_data = self.transform(image) if new_data is None else torch.cat([self.transform(image), new_data],dim=0)
            #print(new_data.shape)
            return new_data
        
        seq, target = self.images[index][:self.sequence_length], self.images[index][self.sequence_length:]

        seque = _transform_time(seq)
        label = _transform_time(target)

       
        yield seque, label
    
    def image_sequence(self, images):
        sequence = []
        sequence_batch = []

        last_action_id = None
        for image in images:
            _, img_name = os.path.split(image)
            action_id, class_id, color_id, frame_id  = img_name.split(".")[0].split("_")
            
            if not action_id == last_action_id and last_action_id != None:
                continue
            sequence_batch.append(img_name)

            if (len(sequence_batch) > self.sequence_length):# or (not action_id == last_action_id and last_action_id != None):
                sequence.append(sequence_batch)
                sequence_batch = []
            
            last_action_id = action_id
        return sequence