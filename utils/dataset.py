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
from torchvision.datasets import VisionDataset

from pycocotools.coco import COCO

from . import utils

class HDF5Dataset(Dataset):
    """ """
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
        """ """
        self.df.close()
        
class ImageDataset(Dataset):
    """Dataset to load images"""
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
    
class SegmentationDataset(VisionDataset):
    """Segmentation dataset for the segmentation lab"""
    def __init__(self, 
                 data_dir: str,
                 transforms = None,
                 transform= None,
                 target_transform = None,
                 img_shape=None,
                 predict=False, 
                 shuffle=True, 
                 augmentation=None
                ):
        super().__init__(data_dir, transforms, transform, target_transform)
        #self.transform = transform
        #self.augmentation = augmentation
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
        
        
    def __getitem__(self, index): # TODO: How do I transform two images at once??
        # Note: Expect (B,H,W) for mask and (B,C,H,W) for image where C is the same size as number of classes
        # Load image path
        image_file = self.images[index]
        
        # Preprocessing
        image = utils.normalize(np.array(Image.open(image_file)))
        image = self.__segment_background(image)
        #image = self.transform(image)

        if not self.predict: 
            gt_image_file = self.labels[index]
            
            # read labels
            gt_image = self.__segment_background(utils.normalize(np.array(Image.open(gt_image_file))))
            #gt_image = self.transform(gt_image)
            
            
            if self.transforms is not None:
                image, gt_image = self.transforms(image, gt_image)
            
            labels = torch.argmax(gt_image, dim=0)
            
            return image.float(), labels
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image.float()
                
    def __len__(self):
        return len(self.images)
    
    def __segment_background(self, img):
        """Create background target label"""
        bkgnd_image =  np.ones(self.img_shape)
        bkgnd_image  = bkgnd_image - img[:,:,0]
        bkgnd_image  = bkgnd_image - img[:,:,1]
        bkgnd_image  = bkgnd_image - img[:,:,2]
   
        img = np.dstack((img,utils.normalize(bkgnd_image)))
        return img

class ClassificationDataset(Dataset):
    """Classification dataset for the classification lab"""
    def __init__(self, data_dir, classes,img_shape=None, transform=None, fineGrained=False, predict=False, shuffle=True):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.img_shape = img_shape
        self.classes = classes
        self.fineGrained = fineGrained
        self.images = images = glob(os.path.join(data_dir, 'image', '*.png'))
        
        assert self.transform != None, "transform cant be empty!"
        
    def __getitem__(self, index:int) -> "tuple":
       
        # Load image path
        image_file = self.images[index]

        image = self.transform(Image.open(image_file))

        # read labels from image_file names
        labels = self.__extract_label(image_file)

        return image, labels
    
    def __len__(self):
        return len(self.images)
    
    def item(self,index) -> "tuple":
        """Get one item from dataset

        :param index: 

        """
        return self.__getitem__(index)
    
    def __extract_label(self, image_file:str) -> str:
        """Extract label from image_file name"""
        #labels = np.zeros(shape=(len(self.classes)), dtype=np.float32)
        path, img_name = os.path.split(image_file)
        names = img_name.split(".")[0].split("_")

        currLabel = names[1] + "_" + names[2] if self.fineGrained else names[1]
        
        if currLabel in self.classes:
            label = self.classes.index(currLabel)
        else:
            raise ValueError("ERROR: Label " + str(currLabel) + " is not defined!")
        
        return label
    
    
    
    
    
class LastFramePredictorDataset(VisionDataset):
    """LastFramePredictor dataset for the GAN lab"""
    def __init__(self, 
                 data_dir: str,
                 transforms = None,
                 transform= None,
                 target_transform = None,
                 img_shape=None,
                 predict=False, 
                 shuffle=True, 
                ):
        super().__init__(data_dir, transforms, transform, target_transform)
        self.data_dir = data_dir
        self.img_shape = img_shape
       
        self.images, self.lastframe = self.image_sequence(sorted(glob(os.path.join(data_dir, '*.png'))))
        
    def __getitem__(self, index):

        image_file = os.path.join(self.data_dir, self.images[index])
        last_image = os.path.join(self.data_dir, self.lastframe[index]) 
        
        image = utils.normalize(np.array(Image.open(image_file)))
    
        gt_image = utils.normalize(np.array(Image.open(last_image)))
        image, gt_image = self.transforms(image, gt_image)
    
       
        return image, gt_image
    
    def image_sequence(self, images):
        """Create an sequence of images and set last image as target.

        :param images: 

        """
        sequence = []
        lastframe = []

        last_frame_id = None
        last_image_name = None
        last_action_id = None
        for image in images:
            _, img_name = os.path.split(image)
            action_id, class_id, color_id, frame_id  = img_name.split(".")[0].split("_")
            
            if last_action_id == None:
                last_action_id = action_id
                last_frame_id = frame_id
                last_image_name = img_name

            
            if action_id != last_action_id:
                lastframe.extend([last_image_name]*(int(last_frame_id)-1)) #*int(last_frame_id)
                #last_action_id = action_id
                #print("length of sets",len(sequence), len(lastframe), last_frame_id, frame_id, last_frame_id)
            else:   
                sequence.append(img_name)
            
            last_action_id = action_id
            last_frame_id = frame_id
            last_image_name = img_name
        
        lastframe.extend([last_image_name]*(int(last_frame_id))) #*int(last_frame_id)
        return sequence, lastframe
    
    def __len__(self):
        return len(self.images)
    
class FutureFramePredictorDataset(VisionDataset):
    """FutureFramePredictorDataset dataset for the RNN lab"""
    def __init__(self, 
                 data_dir: str,
                 sequence_length:int,
                 transforms = None,
                 transform= None,
                 target_transform = None,
                 img_shape=None,
                 predict=False, 
                 shuffle=True, 
                ):
        super().__init__(data_dir, transforms, transform, target_transform)
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.img_shape = img_shape
        self.images = np.array(self.image_sequence(sorted(glob(os.path.join(data_dir, '*.png')))))

    def __getitem__(self, index):
        # Take last image in sequence as target and rest as sequence
        seq, target = (self.images[index][:self.sequence_length], self.images[index][self.sequence_length:])
      
        # Concatenate over channel and transform images
        seque, label = self.transforms(self._to_series(seq),self._to_series(target))
   
        return seque, label
    
    def __len__(self):
        return len(self.images)
    
    def _to_series(self, data):
        """Concatenate images along the channels to create a series
        :param data: 

        """
        return np.dstack([
            utils.normalize(np.asarray(Image.open(os.path.join(self.data_dir, image_file))))
            for image_file in data
        ])
            
    def image_sequence(self, images:list) -> list:
        """Create an sequence of lists with images. The sequence are defined on the sequence_length.

        :param images: list:
        """
        sequence = []
        sequence_batch = []

        last_action_id = None
        last_image_id = None
        for image in images:
            _, img_name = os.path.split(image)
            action_id, class_id, color_id, frame_id  = img_name.split(".")[0].split("_")
            
            # Ensure that last_action_id is not None
            if last_action_id == None:
                last_action_id = action_id
                last_image_id = img_name
                continue
                
            # We want to break the sequence if we start a new image series!
            # Observe that the sequence must be 7 so all sequences below are removed.
            
            sequence_batch.append(last_image_id)

            # Append and start a new sequence if sequence is full: self.sequence_length inputs + 1 target 
            if (len(sequence_batch) > self.sequence_length):# or (not action_id == last_action_id and last_action_id != None):
                sequence.append(sequence_batch)
                sequence_batch = []
            
            # Last image in sequence is added. Reset batch if new action_id starts
            if action_id != last_action_id:
                sequence_batch = []
                
            last_action_id = action_id
            last_image_id = img_name
            
        #sequence_batch.append(last_image_id)  
        return sequence
    
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Image ID
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)
        
        path = self.coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes, areas, labels, iscrowd = ([],[],[],[])
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]['category_id'])
            areas.append(coco_annotation[i]['area'])
            iscrowd.append(coco_annotation[i]['iscrowd'])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.uint8)
        img_id = torch.tensor([img_id])
        areas = torch.tensor(areas, dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.uint8)
        
        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, my_annotation = self.transforms(img,my_annotation)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)