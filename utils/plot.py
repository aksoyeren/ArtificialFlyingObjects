import matplotlib.pyplot as plt
import numpy as np
import torch
import re
import random
import os.path
import scipy.misc
from PIL import Image
from glob import glob
import scipy.io as sio
import torchvision
import itertools
from matplotlib import cm
import math

from . import utils

def data_distribution(imgs, shape=(2,2)):
    f, axs = plt.subplots(*shape, figsize=(10,10))
    axs = axs.flatten()
    
    for idx, ((d,t), ax) in enumerate(zip(imgs, axs)):
        ax.scatter(d[:,0],d[:,1], c=t)
        ax.set_title(f"Plot number: {idx}")
    plt.show()
    
def decision_bondary(model, X = None, Y1 = None, h=0.025):
    model.eval()
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = utils.tensor2numpy(model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()))
    Z = Z.reshape(xx.shape)

    Z[Z>.5] = 1
    Z[Z<= .5] = 0

    Y_pr = model(torch.from_numpy(X).float()).reshape(Y1.shape)
    
    Y = np.copy(Y1)
    Y_pr[Y_pr>.5] = 1
    Y_pr[Y_pr<= .5] = 0
    Y[(Y!=Y_pr) & (Y==0)] = 2
    Y[(Y!=Y_pr) & (Y==1)] = 3
    
    plt.figure()
    #plt.contourf(xx, yy, Z, cmap=plt.cm.PRGn, alpha = .9) 
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    
    
    plt.scatter(X[:, 0][Y==1], X[:, 1][Y==1], marker='+', c='k')
    plt.scatter(X[:, 0][Y==0], X[:, 1][Y==0], marker='o', c='k')
       
    plt.scatter(X[:, 0][Y==3], X[:, 1][Y==3], marker = '+', c='r')   
    plt.scatter(X[:, 0][Y==2], X[:, 1][Y==2], marker = 'o', c='r')
    
    
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.tight_layout()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None: cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
def grid_of_batch(batch):
    # This function can be used to plot the images of one batch! Expects the shape: (batch, channels, height,width)
    grid_img = torchvision.utils.make_grid(batch,pad_value =1)
    plt.imshow(grid_img.permute(1,2,0))
    
def train_vs_validation_loss_and_val_accuracy(train_loss, val_loss, val_accuracy):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    ax1.plot(train_loss, label='train')
    ax1.plot(val_loss, label='valid')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')

    ax2.plot(valid_acc_list, label='valid acc')
    ax2.legend()
    
def stats_class(x = None, y = None, label = 'Training', model = None):
    """
    input :  
             x = input
             y = output
             label = "Provided text string"
             model = the model
             
    output : 
             sensitivity = fraction of correctly classified positive cases
             specificity = fraction of correctly classified negative cases
             accuracy = fraction of correctly classified cases
             loss = typically the cross-entropy error
    """
    
    def binary(y1):
        y1[y1>.5] = 1.
        y1[y1<= .5] = 0.        
        return y1
    
    model.eval()
    
    y_pr = model(torch.from_numpy(x).float()).reshape(y.shape) #, batch_size = x.shape[0], verbose=0
                
    nof_p, tp, nof_n, tn = [np.count_nonzero(k) for k in [y==1, y_pr[y==1.] > 0.5, y==0, y_pr[y==0.]<= 0.5]]
    
    sens = tp / nof_p
    spec = tn / nof_n
    acc = (tp + tn) / (len(y))
    loss = "Not implemented"#model.test(x, y , batch_size =  x.shape[0], verbose=0)
                
    A = ['Accuracy', 'Sensitivity', 'Specificity', 'Loss']
    B = [acc, sens, spec, loss]
    
    print('\n','#'*10,'STATISTICS for {} Data'.format(label), '#'*10, '\n')
    for r in zip(A,B):
         print(*r, sep = '   ')
    return print('\n','#'*50)

def stats_reg(d = None, d_pred = None, label = 'Training', estimat = None):
    
    A = ['MSE', 'CorrCoeff']
    
    pcorr = np.corrcoef(d, d_pred)[1,0]
    
    if label.lower() in ['training', 'trn', 'train']:
        mse = estimat.history['loss'][-1]
    else:
        mse = estimat.history['val_loss'][-1] 

    B = [mse, pcorr]
    
    print(f"\n {'#'*10} STATISTICS for {label} Data {'#'*10}\n")
    for r in zip(A,B):
         print(*r, sep = '   ')
    return print(f"\n {'#'*50}")

def show_statistics(data_folder, fineGrained=False, title="Input Data Statistics"):

    images = glob(os.path.join(data_folder, 'image', '*.png'))
    print(f"\n {'#' * 70}\n{'#'* 21} {title} {'#' * 21} \n{'#' * 70}")
    print(f"total image number \t {len(images)}")

    label_list = []
    label_counter = []
    for i in range(0, len(images)):
        # read labels from image_file names
        path, img_name = os.path.split(images[i])

        names = img_name.split(".")[0].split("_")
        currLabel = f"{names[1]}_{names[2]}" if fineGrained else names[1] #000001_triangle_blue_000001
  
        if np.isin(currLabel, label_list):
            loc = label_list.index(currLabel)
            label_counter[loc] += 1  # remove the class unknown
        else:
            label_list.append(currLabel)
            label_counter.append(1)

    print(f"total class number \t {len(label_list)}")
    for i in range(0, len(label_list)):
        print(f"class {label_list[i]} \t {label_counter[i]} images")

    print("#" * 70)

class Segmentation():
    
    @staticmethod
    def data(X, Y, img_nbr_toshow=4, nrow=5):
        # show testing results
        fig, axes = plt.subplots(2,1,figsize=(10, 5))
        fig.suptitle('Data Samples', size=20)
        flatten_axes = axes.flatten()

        for i, (image, title) in enumerate([
            (X, f"Input images {0}-{img_nbr_toshow}"),
            (add_rgb2tensor(Y), f"Truth images {0}-{img_nbr_toshow}")
        ]):

            grid_image = torchvision.utils.make_grid(image[:img_nbr_toshow], nrow=nrow,pad_value=0.5).numpy().transpose(1,2,0)
            #image = utils.normalize(image)
            show(grid_image,ax=axes[i])
            flatten_axes[i].set_xticks([])
            flatten_axes[i].set_yticks([])
            flatten_axes[i].grid(False)
            flatten_axes[i].set_xlabel(title, size=20)

        plt.show()
        
    @staticmethod
    def results(X, Y, predictions, num_classes, img_nbr_toshow=4, nrow=5):

        fig, axes = plt.subplots(3,1,figsize=(10, 10))
        fig.suptitle(f"Sample Segmentation Results", size=20)
        flatten_axes = axes.flatten()

        for i, (image, title) in enumerate([
            (X, f"Input images {0}-{img_nbr_toshow}"),
            (add_rgb2tensor(Y,num_classes), f"Truth images {0}-{img_nbr_toshow}"),
            (add_rgb2tensor(predictions,num_classes), f"Predicted images {0}-{img_nbr_toshow}"),
        ]):
            grid_image = torchvision.utils.make_grid(image[:img_nbr_toshow], nrow=nrow,pad_value=0.5).numpy().transpose(1,2,0)
            show(grid_image,ax=axes[i])
            flatten_axes[i].set_xticks([])
            flatten_axes[i].set_yticks([])
            flatten_axes[i].grid(False)
            flatten_axes[i].set_xlabel(title, size=20)
        plt.tight_layout()
        plt.show()

class Classification:
    @staticmethod
    def data(X:"tensor", Y:"tensor"=None, **plot_kwargs):
        X = X.permute(0,3,2,1)

        image_with_labels(X,Y, "Classification samples",**plot_kwargs)
    
    @staticmethod
    def results(X, predictions,Y=None, nimages=4, nrow=5,**plot_kwargs):
        X = X.permute(0,3,2,1)

        image_with_labels(X, Y, "Classification sampled results",**plot_kwargs)

        image_with_labels(X,predictions, "Classification sampled predictions")

def convert_4Dimgbatch_2_3Dimgbatch(inputImgBatch:torch.tensor):
    "Expects shape [N,H,W,C]"
    # generate one channel labelled image as ground truth
    color_map = {0: np.array([1, 0, 0]),
                 1: np.array([0, 1, 0]),
                 2: np.array([0, 0, 1]),
                 3: np.array([0, 0, 0])}

    color_img = np.empty((*inputImgBatch.shape[:-1],3), dtype=np.float32)
    for f in range(0, len(color_img)):

        curr_pred = inputImgBatch[f, :, :, :]

        maxImg =np.argmax(curr_pred, axis=2)

        for color, value in color_map.items():
            color_img[f,maxImg==color] = value
    return color_img



def image_with_labels(data:"tensor", labels:"tensor"=None, title:str=None, nimages:int=10, nrows:int=2, fig_dimension=1) -> None:
    
    if len(data)< nimages:
        nimages = len(data)
 
    columns = math.ceil(nimages/nrows)
    
    if nrows*columns > nimages:
        nrows = math.ceil(nimages/columns)
    
    fig = plt.figure(figsize=(fig_dimension*columns,1.4*fig_dimension*nrows))

    for i in range(1, nimages + 1):
        ax = fig.add_subplot(nrows, columns, i)
        ax.imshow(data[i])
        ax.set_xlabel(f"Label: {labels[i]}") if labels is not None else None
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    if labels is None:
        fig.suptitle(title,x=0.45, y=.95) 
        
        fig.subplots_adjust(
            left=0,
            right=0.9,
            top=0.9,
            bottom=0,
            wspace=0.1,
            hspace=-0.45
        )
    else:
        fig.suptitle(title) #,x=0.45, y=.95
        
        fig.subplots_adjust(
            #left=0,
            #right=1,
            top=0.9,#+((nrows-1)*0.045),
            #bottom=0,
            wspace=0,
            #hspace=0
        )
        
    #plt.tight_layout(h_pad=0,w_pad=0)
    fig.tight_layout(pad=0, h_pad=0,w_pad=0)
    plt.show()
    
def show(img, ax=None, cmap=None): 
    #np_image = np.transpose(img, (1,2,0))
    if ax:
        ax.imshow(img, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(img, interpolation='nearest', cmap=cmap)
        
def add_rgb2tensor(image,num_classes=4):
    assert len(image.shape) == 3, f"Wrong image shape! Expected shape of 3! Got: Image length: {len(image.shape)}"
    cmap_vol = np.apply_along_axis(cm.get_cmap(lut=num_classes), 0, image)
    cmap_vol = np.delete(cmap_vol, 3, 1) # Delete alpha
    cmap_vol = torch.from_numpy(cmap_vol)
    
    
    return cmap_vol

def rgb2index(one_hot_vector):
    return np.argmax(one_hot_vector, axis=1)
    