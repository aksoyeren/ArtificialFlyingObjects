from sklearn import preprocessing
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import argparse

def label_encoder(labels:list)  -> "tensor":
    """Convert list of string labels to tensors

    :param labels:

    """
    

    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
                  
    return torch.as_tensor(targets)

def tensor2numpy(data:"tensor") -> "tensor":
    """Send to CPU. If computational graph is connected then detach it as well.

    :param data:"tensor":


    """
    if data.requires_grad:
        return data.detach().cpu().numpy()
    else:
        return data.cpu().numpy()
    
def label2rgb(labels:"tensor") -> "tensor":
    """Convert numerical labels to one-hot encoded vector

    :param labels:"tensor": 

    """
    return torch.nn.functional.one_hot(labels)

def onehot2int(labels:"np.ndarray") -> "np.ndarray":
    """Convert a numpy one-hot vector to integer labels.

    :param labels:"np.ndarray": 

    """
    return np.argmax(labels, axis=0)

def normalize(x:"tensor|np.ndarray") -> "tensor|np.ndarray":
    """Min-max normalization (0-1):

    :param x:"tensor|np.ndarray": 
    :returns: Union[Tensor,np.ndarray] - Return same type as input but scaled between 0 - 1

    """
    return (x - x.min())/(x.max()-x.min())

def normalize_01(inp: "np.ndarray") -> "np.ndarray":
    """Squash image input to the value range [0, 1] (no clipping)

    :param inp: np.ndarray:

    """
    inp_out = (inp - np.min(inp)) / np.ptp(inp)
    return inp_out

def dict_to_args(d:dict) -> "argparse.Namespace":
    """Convert a recursive dictionary to namespace.

    :param d:dict: 

    """
    def dict_to_args_recursive(d:"dict|int|str", prefix='')->"argparse.Namespace":
        """Recursively apply for each input d.
        
        :param d:"dict|int|str": 
        :param prefix: Default value = '')
        """
        args = argparse.Namespace()
        for k, v in d.items():
            if type(v) == dict:
                args.__setattr__(k, dict_to_args_recursive(v, prefix=k))
            elif type(v) in [tuple, list]:
                continue
            
            else:
                args.__setattr__(k, v)
        return args
            
    return dict_to_args_recursive(d)