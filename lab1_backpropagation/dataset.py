import torchvision.transforms as transforms
import os

class HDF5Dataset(Dataset):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def __init__(self, h5_path=None):
        assert os.path.isdir(h5_path), "Path to files does not exist!"
        
        self.train = test_catvnoncat.h5(h5_path, 'r')
        self.length = 

    def __getitem__(self, index): #to enable indexing
        record = self.train[str(index)]
        image = record['X'].value

        # transform to PIL image
        image = Image.fromarray(pixels.astype('uint8'), 'RGB') # assume your data is  uint8 rgb
        label = record['y'].value

        # transformation here
        # torchvision PIL transformations accepts one image as input
        image = self.transform(image)
        return (
                image,
                label,
        )

    def __len__(self):
        return len(h5py.File(h5_path, 'r')) 