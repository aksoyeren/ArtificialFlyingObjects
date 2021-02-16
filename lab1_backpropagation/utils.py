from sklearn import preprocessing
import torch
import torchvision
import matplotlib.pyplot as plt

def label_encoder(labels):
    "Convert list of string labels to tensors"

    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
                  
    return torch.as_tensor(targets)

def plot_grid_of_batch(batch):
    # This function can be used to plot the images of one batch! Expects the shape: (batch, channels, height,width)
    grid_img = torchvision.utils.make_grid(batch,pad_value =1)
    plt.imshow(grid_img.permute(1,2,0))
    
def plot_train_vs_validation_loss_and_val_accuracy(train_loss, val_loss, val_accuracy):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    ax1.plot(train_loss, label='train')
    ax1.plot(val_loss, label='valid')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')

    ax2.plot(valid_acc_list, label='valid acc')
    ax2.legend()