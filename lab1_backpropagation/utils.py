import h5py
from typing import List, Tuple, TypeVar

T = TypeVar('T')  # Can be anything
S = TypeVar('S', str, bytes)  
F = TypeVar('F', int, float) 

class DataH5:
    training_data_dir = "../data/FlyingObjectDataset_10K/training"
    validation_data_dir = "../data/FlyingObjectDataset_10K/validation"
    testing_data_dir = "../data/FlyingObjectDataset_10K/testing"
    
    def __init__(self, img_shape:Tuple[F, F, F], classes:List[T], path='flyingData.hdf5'):
        self.f = h5py.File(path, 'w')
        self.img_shape = img_shape
        self.img_classes = classes
    
    def create_dataset(self, dir_path:S):
        imgs = self.get_img_paths(dir_path)
        grp = self.create_group(imgs, dir_path.rsplit("/",1)[1])
        
        self.images_to_group(grp, imgs)
        
    def create_group(self, imgs:List[T],group_name:S):
        
        num_imgs = len(imgs)
        data_shape = (num_imgs, *self.img_shape )
        label_shape = (num_imgs, len(self.img_classes))
        
        grp = self.f.create_group(group_name)
        grp.create_dataset('data', data_shape, np.float)
        grp.create_dataset('label', label_shape, np.float)
        
        
        return grp

    def images_to_group(self,group, imgs):
        for i, img_path in enumerate(imgs):
            print("\rProcessing %i" % i)
            image = resize(imageio.imread(image_path), self.img_shape)
            
            labels = np.zeros(shape=(len(self.img_classes)), dtype=np.float32)
                       
            img_path, img_name = os.path.split(img_path)
            fn, ext = img_name.split(".")
            names = fn.split("_")
                       
            currLabel = names[1] + "_" + names[2]
            if np.isin(currLabel, self.img_classes):
                loc = self.img_classes.index(currLabel)
                labels[loc] = 1
            else:
                print("ERROR: Label " + str(currLabel) + " is not defined!")
        
        
            group["data"][i, ...] = image
            group["label"][i, ...] = labels

    def get_img_paths(self, dir_path:S):
        return glob(os.path.join(dir_path, 'image', '*.png'))
    
    def get(self):
        pass
    
    def get_groups(self):
        return [name for name in f]

    
    
def image_dir_to_h5(dir_path, set, output_file):
    images = glob(os.path.join(dir_path, 'image', '*.png'))
    n_image = len(images)
    cfg = flying_objects_config()
    


    output_file.create_dataset(set + '_x', set_shape, np.float)
    output_file.create_dataset(set + '_y', (n_image, len(cfg.CLASSES)), np.float)


    for i, image_path in enumerate(images):
        print("\rProcessing %i" % i)
        image = resize(imageio.imread(image_path), set_shape[1:])


        labels = np.zeros(shape=(len(cfg.CLASSES)), dtype=np.float32)
        path, img_name = os.path.split(image_path)
        fn, ext = img_name.split(".")
        names = fn.split("_")
        currLabel = names[1] + "_" + names[2]
        if np.isin(currLabel, cfg.CLASSES):
            loc = cfg.CLASSES.index(currLabel)
            labels[loc] = 1
        else:
            print("ERROR: Label " + str(currLabel) + " is not defined!")

        output_file[set + '_x'][i, ...] = image
        output_file[set + '_y'][i, ...] = labels