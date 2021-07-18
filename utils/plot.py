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

def data_distribution(imgs:"list|dict", shape=(2,2)) -> None:
    """Plot scatter distribution for a list of images.

    :param imgs:"list|dict": 
    :param shape:  (Default value = (2)
    :param 2): 

    """
    f, axs = plt.subplots(*shape, figsize=(10,10))
    axs = axs.flatten()
    
    if isinstance(imgs,list):
        for idx, ((d,t), ax) in enumerate(zip(imgs, axs)):
            ax.scatter(d[:,0],d[:,1], c=t)
            ax.set_title(f"Plot number: {idx}")
    elif isinstance(imgs,dict):
        for (key, (d,t)), ax in zip(imgs.items(), axs):
            ax.scatter(d[:,0],d[:,1], c=t)
            ax.set_title(key)
    plt.show()
    
def decision_bondary(model:"torch.nn.Module", X:"tensor"=None, Y1:"tensor"=None, h:float=0.025) -> None:
    """For a model, plot the decision boundary based on the input X and Y1

    :param model:"torch.nn.Module": 
    :param X:"tensor":  (Default value = None)
    :param Y1:"tensor":  (Default value = None)
    :param h:float:  (Default value = 0.025)

    """
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


def confusion_matrix(cm:"tensor",
                          target_names:list,
                          title='Confusion matrix',
                          cmap:str=None,
                          normalize:bool=True):
    """Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    :param cm:"tensor": 
    :param target_names:list: 
    :param title:  (Default value = 'Confusion matrix')
    :param cmap:str:  (Default value = None)
    :param normalize:bool:  (Default value = True)

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
    
def train_vs_validation_loss_and_val_accuracy(train_loss:list, val_loss:list, val_accuracy:list) -> None:
    """Plot the training vs validation loss and validation accuracy

    :param train_loss:list: 
    :param val_loss:list: 
    :param val_accuracy:list: 

    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    ax1.plot(train_loss, label='train')
    ax1.plot(val_loss, label='valid')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')

    ax2.plot(valid_acc_list, label='valid acc')
    ax2.legend()
    
def stats_class(x:list=None, y:list=None, label:str='Training', model:"nn.Module"=None)-> None:
    """

    :param x:list:  (Default value = None)
    :param y:list:  (Default value = None)
    :param label:str:  (Default value = 'Training')
    :param model:"nn.Module":  (Default value = None)
    
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
        """

        :param y1: 

        """
        y1[y1>.5] = 1.
        y1[y1<= .5] = 0.        
        return y1
    
    model.eval()
    
    y_pr = model(torch.from_numpy(x).float()).reshape(y.shape) #, batch_size = x.shape[0], verbose=0
                
    nof_p, tp, nof_n, tn = [np.count_nonzero(k) for k in [y==1, y_pr[y==1.] > 0.5, y==0, y_pr[y==0.]<= 0.5]]
    
    sens = tp / nof_p
    spec = tn / nof_n
    acc = (tp + tn) / (len(y))
    #loss = "Not implemented"#model.test(x, y , batch_size =  x.shape[0], verbose=0)
                
    A = ['Accuracy', 'Sensitivity', 'Specificity']#, 'Loss']
    B = [acc, sens, spec]#, loss]
    
    print('\n','#'*10,'STATISTICS for {} Data'.format(label), '#'*10, '\n')
    for r in zip(A,B):
         print(*r, sep = '   ')
    print('\n','#'*50)

def stats_reg(d:list=None, d_pred:list=None, label:str='Training', estimat:"nn.Module" = None):
    """Print stats for regression.
    
    **Note:** Not compatible with torch.nn.Module!

    :param d:list:  (Default value = None)
    :param d_pred:list:  (Default value = None)
    :param label:str:  (Default value = 'Training')
    :param estimat:"nn.Module":  (Default value = None)

    """
    
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

def data_statistics(data_folder:str, fineGrained:bool=False, title:str="Input Data Statistics") -> None:
    """Show statistics of data in folder.

    :param data_folder:str: 
    :param fineGrained:bool:  (Default value = False)
    :param title:str:  (Default value = "Input Data Statistics")

    """

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
    """Plot functions for Segmentation."""
    
    @staticmethod
    def data(X:"tensor", Y:"tensor"=None, **plot_kwargs:dict) -> None:
        """Plot input images and target images.

        :param X:"tensor": 
        :param Y:"tensor":  (Default value = None)
        :param **plot_kwargs:dict: 

        """
        X = X.permute(0,2,3,1)
        Y = Y.unsqueeze(1).permute(0,2,3,1)
        image_with_labels(X, title="Segmentation Input",**plot_kwargs)
        image_with_labels(Y, title="Segmentation Target",**plot_kwargs)
        
    @staticmethod
    def results(X:"tensor", predictions:"tensor",Y:"tensor"=None, nimages:int=4, nrow:int=5,**plot_kwargs:dict)  -> None:
        """Plot input images, target images and result images of the predictions.

        :param X:"tensor": 
        :param predictions:"tensor": 
        :param Y:"tensor":  (Default value = None)
        :param nimages:int:  (Default value = 4)
        :param nrow:int:  (Default value = 5)
        :param **plot_kwargs:dict: 

        """
        X = X.permute(0,2,3,1)
        Y = Y.unsqueeze(1).permute(0,2,3,1)
        predictions = predictions.unsqueeze(1).permute(0,2,3,1)
        image_with_labels(X, title="Segmentation Input",**plot_kwargs)
        image_with_labels(Y, title="Segmentation Target",**plot_kwargs)
        image_with_labels(predictions, title="Segmentation Prediction",**plot_kwargs)

class GAN():
    """Plot functions for Segmentation."""
    
    @staticmethod
    def data(X:"tensor", Y:"tensor"=None, **plot_kwargs:dict) -> None:
        """Plot input images and target images.

        :param X:"tensor": 
        :param Y:"tensor":  (Default value = None)
        :param **plot_kwargs:dict: 

        """
        X = X.permute(0,2,3,1)
        Y = Y.permute(0,2,3,1)
        image_with_labels(X, title="Input",**plot_kwargs)
        image_with_labels(Y, title="Target",**plot_kwargs)
        
    @staticmethod
    def results(X:"tensor", predictions:"tensor",Y:"tensor"=None, nimages:int=4, nrow:int=5,**plot_kwargs:dict)  -> None:
        """Plot input images, target images and result images of the predictions.

        :param X:"tensor": 
        :param predictions:"tensor": 
        :param Y:"tensor":  (Default value = None)
        :param nimages:int:  (Default value = 4)
        :param nrow:int:  (Default value = 5)
        :param **plot_kwargs:dict: 

        """
        X = X.permute(0,2,3,1)
        Y = Y.permute(0,2,3,1)
        predictions = predictions.permute(0,2,3,1)
        image_with_labels(X, title="Input",**plot_kwargs)
        image_with_labels(Y, title="Target",**plot_kwargs)
        image_with_labels(predictions, title="Predictions",**plot_kwargs)

class RNN():
    """Plot functions for Segmentation."""
    
    @staticmethod
    def data(
        X:"tensor", 
        Y:"tensor", 
        sequence_len:int, 
        nrows:int=3, 
        title="Input/Target", 
        **plot_kwargs:dict
    ) -> None:
        """Plot input images and target images.
        nimages for image_with_labels are set based on the number of rows and sequence length.
        
        :param X:"tensor": Input image with format: B,C,W,H 
        :param Y:"tensor":  Target image with format: B,C,W,H (Default value = None)
        :param **plot_kwargs:dict: Extra parameters for image_with_labels function

        """
     
        channels = 3
        # Split on channels and stack (B,S,C,W,H)
        images = torch.stack(torch.split(X,channels,dim=1),dim=1)
        targets = torch.stack(torch.split(Y,channels,dim=1),dim=1)
        
        batch_size= images.shape[0]
        width = images.shape[3]
        height = images.shape[4]

        # Merge input and target (B,C,W,H)
        image_with_targets = torch.cat(
            (images,targets),dim=1
        ).view(batch_size*(sequence_len+1), channels,width,height).permute(0,2,3,1)
        
        # Apply label for each image/target in sequence
        labels = [
            "Target" if i%(sequence_len+1) == 0 and i != 0 else f"Serie {i%(sequence_len+1)}" 
            for i in range(1,image_with_targets.shape[0]*(sequence_len+1))
        ]
        
        image_with_labels(
            image_with_targets, 
            labels=labels,
            label_prefix="",
            title=title, 
            nrows=nrows, 
            nimages=(sequence_len+1)*nrows, 
            fig_dimension=0.8,
            **plot_kwargs
        )
        
    @staticmethod
    def results(
        X:"tensor", 
        predictions:"tensor",
        sequence_len:int, 
        Y:"tensor"=None,  
        nrows:int=2,
        **plot_kwargs:dict
    ) -> None:
        """Plot input images, target images and result images of the predictions.

        :param X:"tensor": 
        :param predictions:"tensor": 
        :param Y:"tensor":  (Default value = None)
        :param nimages:int:  (Default value = 4)
        :param nrow:int:  (Default value = 5)
        :param **plot_kwargs:dict: 

        """
        if Y is not None:
            RNN.data(X,Y, sequence_len, nrows=nrows)
        RNN.data(X,predictions, sequence_len, nrows=nrows, title="Input/Prediction")
    
    @staticmethod   
    def diff(
        X:"tensor",
        Y:"tensor", 
        nrows:int=2,
        diff_color:str="orange",
        **plot_kwargs:dict
    ):
        """Indicate the difference in the input images with the target with orange color."""
        channels=3
        X = torch.stack(torch.split(X,channels,dim=1),dim=1)
        Y = torch.stack(torch.split(Y,channels,dim=1),dim=1)
        
        batch_size= X.shape[0]
        width = X.shape[3]
        height = X.shape[4]
        sequence_len = X.shape[1]
   
        mask =  torch.cat([X,Y],1) - Y
        mask[mask <= 0] = 0
        mask = mask.view(batch_size*(sequence_len+1), channels,width,height)
        image_mask = (255*torch.cat([X, Y],1).view(
            batch_size*(sequence_len+1), channels,width,height
        )).type(torch.uint8)

        diff = []
        for x,y in zip(image_mask, mask):
            _, masks = torch.unique(y,return_inverse=True)

            masks = masks.type(torch.bool) if masks.max()==0 else torch.sum(masks,0).type(torch.bool)

            diff.append(torchvision.utils.draw_segmentation_masks(
                x, 
                masks,
                alpha=0.9,

                colors=[diff_color]*len(masks)
            ))
        
        diff = torch.stack(diff).permute(0,3,2,1)
        
        labels = [
            "Target" if i%(sequence_len+1) == 0 and i != 0 else f"Serie {i%(sequence_len+1)}" 
            for i in range(1,diff.shape[0])
        ]
        image_with_labels(
            diff, 
            labels=labels,
            label_prefix="",
            nimages=(sequence_len+1)*nrows,
            nrows=nrows,
            title="Diff (orange color) inputs with target",
            **plot_kwargs
        )
        
class Classification:
    """Plot functions for Classification."""
    @staticmethod
    def data(X:"tensor", Y:"tensor"=None, **plot_kwargs:dict) -> None:
        """Plot input images and target labels.

        :param X:"tensor": 
        :param Y:"tensor":  (Default value = None)
        :param **plot_kwargs:dict: 

        """
        X = X.permute(0,2,3,1)

        image_with_labels(X,Y, "Classification samples",**plot_kwargs)
    
    @staticmethod
    def results(X:"tensor", predictions:"tensor",Y:"tensor"=None, nimages:int=4, nrow:int=5,**plot_kwargs:dict) -> None:
        """Plot image input, target labels and result labels of the predictions.

        :param X:"tensor": 
        :param predictions:"tensor": 
        :param Y:"tensor":  (Default value = None)
        :param nimages:int:  (Default value = 4)
        :param nrow:int:  (Default value = 5)
        :param **plot_kwargs:dict: 

        """
        X = X.permute(0,2,3,1)

        image_with_labels(X, Y, "Classification Input",**plot_kwargs)

        image_with_labels(X,predictions, "Classification Predictions",**plot_kwargs)

class LastFrame:
    """Plot functions for LastFramePredictions."""
    @staticmethod
    def data(X:"tensor", Y:"tensor"=None, **plot_kwargs:dict) -> None:
        """Plot input images and target images.

        :param X:"tensor": 
        :param Y:"tensor":  (Default value = None)
        :param **plot_kwargs:dict: 

        """
        X = X.permute(0,2,3,1)
        Y = Y.unsqueeze(1).permute(0,2,3,1)
        image_with_labels(X, title="Segmentation Input",**plot_kwargs)
        image_with_labels(Y, title="Segmentation Target",**plot_kwargs)
    
    @staticmethod
    def results(X:"tensor", predictions:"tensor",Y:"tensor"=None, nimages:int=4, nrow:int=5,**plot_kwargs:dict) -> None:
        """Plot image input, target images and result images of the predictions.

        :param X:"tensor": 
        :param predictions:"tensor": 
        :param Y:"tensor":  (Default value = None)
        :param nimages:int:  (Default value = 4)
        :param nrow:int:  (Default value = 5)
        :param **plot_kwargs:dict: 

        """
        X = X.permute(0,2,3,1)
        Y = Y.unsqueeze(1).permute(0,2,3,1)
        predictions = predictions.unsqueeze(1).permute(0,2,3,1)
        image_with_labels(X, title="Segmentation Input",**plot_kwargs)
        image_with_labels(Y, title="Segmentation Target",**plot_kwargs)
        image_with_labels(predictions, title="Segmentation Prediction",**plot_kwargs)
        
class FutureFrame:
    """Plot functions for FutureFramePredictions."""
    @staticmethod
    def data(X:"tensor", Y:"tensor"=None, **plot_kwargs:dict) -> None:
        """Plot input images and target images.

        :param X: tensor":
        :param Y: tensor":  (Default value = None)
        :param X:"tensor": 
        :param Y:"tensor":  (Default value = None)
        :param **plot_kwargs:dict: 

        """
        X = X.permute(0,2,3,1)
        Y = Y.unsqueeze(1).permute(0,2,3,1)
        image_with_labels(X, title="Segmentation Input",**plot_kwargs)
        image_with_labels(Y, title="Segmentation Target",**plot_kwargs)
    
    @staticmethod
    def results(X, predictions,Y=None, nimages=4, nrow=5,**plot_kwargs):
        """Plot image input, target images and result images of the predictions.

        :param X: param predictions:
        :param Y: Default value = None)
        :param nimages: Default value = 4)
        :param nrow: Default value = 5)
        :param predictions: param **plot_kwargs:
        :param **plot_kwargs: 

        """
        X = X.permute(0,2,3,1)
        Y = Y.unsqueeze(1).permute(0,2,3,1)
        predictions = predictions.unsqueeze(1).permute(0,2,3,1)
        image_with_labels(X, title="Segmentation Input",**plot_kwargs)
        image_with_labels(Y, title="Segmentation Target",**plot_kwargs)
        image_with_labels(predictions, title="Segmentation Prediction",**plot_kwargs)
    
class Detection:
    
    @staticmethod
    def __create_palette(N:int):
        import seaborn as sns
        palette = sns.color_palette("bright", N)
        return palette
    
    @staticmethod
    def __labels2rgb(labels:list,pal):
        return map(lambda x: tuple(map(lambda x: int(x*255), pal[x])), labels)
    
    @staticmethod
    def __draw_bbox(images:"tensor", annotations:"tensor",num_categories:int, **kwargs) -> "tensor":
        labels = torch.tensor([j for i in annotations for j in i['labels']])
        palette = Detection.__create_palette(num_categories)
        imgs = []
        for (image, annot) in zip(images,annotations):
            labels = list(map(lambda x: x.item(), annot['labels']))
            imgs.append(
                torchvision.utils.draw_bounding_boxes(
                    (image*255).type(torch.uint8),
                    annot["boxes"], 
                    labels=list(map(str,labels)), 
                    colors=list(Detection.__labels2rgb(labels, palette)), 
                    **kwargs
                ).permute(1,2,0)
            )
        return torch.stack(imgs)
    @staticmethod
    def data(X:"tensor",annotations:"tensor", num_categories:int, plot_kwargs:dict, **draw_bbox) -> None:
        """Plot input images and target bboxes."""
        image_with_labels(Detection.__draw_bbox(X,annotations, num_categories, **draw_bbox),fig_dimension=2, title="Detection",**plot_kwargs)
        
    @staticmethod
    def results(X:"tensor",predictions:"tensor",annotations:"tensor", num_categories:int, plot_kwargs:dict, **draw_bbox) -> None:
        """Plot image input with target bboxes and result bboxes of the predictions."""
        
        image_with_labels(Detection.__draw_bbox(X,predictions, num_categories, **draw_bbox),fig_dimension=2, title="Predictions",**plot_kwargs)
        image_with_labels(Detection.__draw_bbox(X,annotations, num_categories, **draw_bbox),fig_dimension=2, title="Expected",**plot_kwargs)
        
        
def convert_4Dimgbatch_2_3Dimgbatch(inputImgBatch:torch.tensor):
    """Convert a batch of images from 4D to 3D. Maps the input labels to RGB colors.

    :param inputImgBatch: torch.tensor:
    :param inputImgBatch:torch.tensor: 

    """
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

def torch_grid(X,Y,predictions,num_classes=10, nimages=10):
    """Create a grid of images for X, Y and predictions.
    
    **NOTE:** Depricated in favor of image_with_labels.

    :param X: param Y:
    :param predictions: param num_classes:  (Default value = 10)
    :param nimages: Default value = 10)
    :param Y: param num_classes:  (Default value = 10)
    :param num_classes: Default value = 10)

    """
    fig, axes = plt.subplots(3,1,figsize=(10, 10))
    fig.suptitle(f"Sample Segmentation Results", size=20)
    flatten_axes = axes.flatten()

    for i, (image, title) in enumerate([
        (X, f"Input images {0}-{nimages}"),
        (add_rgb2tensor(Y,num_classes), f"Truth images {0}-{nimages}"),
        (add_rgb2tensor(predictions,num_classes), f"Predicted images {0}-{nimages}"),
    ]):
        grid_image = torchvision.utils.make_grid(image[:nimages], nrow=nrow,pad_value=0.5).numpy().transpose(1,2,0)
        show(grid_image,ax=axes[i])
        flatten_axes[i].set_xticks([])
        flatten_axes[i].set_yticks([])
        flatten_axes[i].grid(False)
        flatten_axes[i].set_xlabel(title, size=20)
    plt.tight_layout()
    plt.show()

def image_with_labels(data:"tensor", labels:"tensor"=None, title:str=None, nimages:int=10, nrows:int=2, fig_dimension=1,title_size=10, label_prefix="Label: ") -> None:
    """Creates a grid of images with/without labels.

    :param data: tensor": B,W,H,C
    :param labels: tensor":  (Default value = None)
    :param title: str:  (Default value = None)
    :param nimages: int:  (Default value = 10)
    :param nrows: int:  (Default value = 2)
    :param fig_dimension: Default value = 1)
    :param data:"tensor": 
    :param labels:"tensor":  (Default value = None)
    :param title:str:  (Default value = None)
    :param nimages:int:  (Default value = 10)
    :param nrows:int:  (Default value = 2)

    """
    image_ratio = data[0].shape[0] /data[0].shape[1]
    if len(data)< nimages:
        nimages = len(data)
 
    columns = math.ceil(nimages/nrows)
    
    if nrows*columns > nimages:
        nrows = math.ceil(nimages/columns)
    
    fig = plt.figure(figsize=(fig_dimension*columns,1.4*fig_dimension*nrows*image_ratio))
    for i in range(0, nimages):
        ax = fig.add_subplot(nrows, columns, i+1)
        ax.imshow(data[i])
        ax.set_xlabel(f"{label_prefix}{labels[i]}") if labels is not None else None
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    if labels is None:
        fig.suptitle(title,x=0.5, y=.95, size=title_size) 
        
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
    
def show(img:"np.ndarray", ax=None, cmap:str=None) -> None: 
    """Show plot for image *img*

    :param img: 
    :param ax: Default value = None)
    :param img:"np.ndarray": 
    :param cmap:str:  (Default value = None)

    """
    #np_image = np.transpose(img, (1,2,0))
    if ax:
        ax.imshow(img, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(img, interpolation='nearest', cmap=cmap)
        
def add_rgb2tensor(image:"tensor",num_classes:int=4) -> "tensor":
    """Add RGB colors to tensor with predefined length of num_classes

    :param image: 
    :param num_classes: Default value = 4)


    """
    assert len(image.shape) == 3, f"Wrong image shape! Expected shape of 3! Got: Image length: {len(image.shape)}"
    cmap_vol = np.apply_along_axis(cm.get_cmap(lut=num_classes), 0, image)
    cmap_vol = np.delete(cmap_vol, 3, 1) # Delete alpha
    cmap_vol = torch.from_numpy(cmap_vol)
    
    
    return cmap_vol

def rgb2index(one_hot_vector:"np.ndarray") -> "np.ndarray":
    """Convert a RGB vector to label.
    :param one_hot_vector:"np.ndarray": 

    """
    return np.argmax(one_hot_vector, axis=1)
    