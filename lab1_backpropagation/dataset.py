import h5py
from pathlib import Path
import torch
from torch.utils.data import Dataset,DataLoader
from easydict import EasyDict as edict


class CustomDataLoader:
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
        assert(p.is_file())
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
