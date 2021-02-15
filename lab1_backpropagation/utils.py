from sklearn import preprocessing
import torch
def label_encoder(labels):
    "Convert list of string labels to tensors"
    
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
                  
    return torch.as_tensor(targets)
