import re
import random
import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy.misc
from PIL import Image
from glob import glob
import scipy.io as sio

def get_dataset_size(data_folder):
    images = glob(os.path.join(data_folder, 'image', '*.png'))
    return len(images)

def generate_classification_batches(data_folder, image_shape, batch_size, classes, fineGrained=False):
    images = glob(os.path.join(data_folder, 'image', '*.png'))
    n_image = len(images)

    # this line is just to make the generator infinite, keras needs that
    while True:

        # Randomize the indices to make an array
        indices_arr = np.random.permutation(n_image)
        for batch in range(0, len(indices_arr), batch_size):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + batch_size)]

            # initializing the arrays, x_train and y_train
            x_train = []  # np.empty([0, image_shape[0],image_shape[1],image_shape[2]], dtype=np.float32)
            y_train = []

            for i in current_batch:

                image_file = images[i]
                image = Image.open(image_file)
                image = image.resize((image_shape[0], image_shape[1]))
                image = np.asarray(image)

                # read labels from image_file names
                labels = np.zeros(shape=(len(classes)), dtype=np.float32)
                path, img_name = os.path.split(image_file)
                fn, ext = img_name.split(".")
                names = fn.split("_")

                if fineGrained:
                    currLabel = names[1] + "_" + names[2]
                else:
                    currLabel = names[1]  # 000001_triangle_blue_000001

                if np.isin(currLabel, classes):
                    loc = classes.index(currLabel)
                    labels[loc] = 1
                else:
                    print("ERROR: Label " + str(currLabel) + " is not defined!")

                # Appending them to existing batch
                x_train.append(image)  # x_train = np.append(x_train, [image], axis=0)
                y_train.append(labels)
            # y_train = to_categorical(y_train, num_classes=len(classes))

            batch_images = np.array(x_train)
            batch_lables = np.array(y_train)
            # normalize image data (not the labels)
            # batch_images = batch_images.astype('float32') / 255

            yield (batch_images, batch_lables)

def generate_augmented_classification_batches(in_gen, image_gen):
    for data, labels in in_gen:
        aug_data = image_gen.flow(255 * data, labels, batch_size=data.shape[0])

        aug_img, aug_lab = next(aug_data)

        yield aug_img / 255.0, aug_lab

def generate_segmentation_batches(data_folder, image_shape, batch_size):

    images = glob(os.path.join(data_folder, 'image', '*.png'))
    n_image = len(images)

    labels = {
        re.sub(r'gt_', '', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image', 'gt_*.png'))}
    n_labels = len(labels)

    assert n_labels == n_image, "numbers of image and ground truth labels are not the same>> image nbr: %d gt nbr: %d"  % (n_image, n_labels)

    # this line is just to make the generator infinite, keras needs that
    while True:

        # Randomize the indices to make an array
        indices_arr = np.random.permutation(n_image)
        for batch in range(0, len(indices_arr), batch_size):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + batch_size)]

            # initializing the arrays, x_train and y_train
            x_train = []  # np.empty([0, image_shape[0],image_shape[1],image_shape[2]], dtype=np.float32)
            y_train = []

            for i in current_batch:

                image_file = images[i]
                image = Image.open(image_file)
                image = image.resize((image_shape[0], image_shape[1]))
                image = np.asarray(image)
                # read labels
                gt_image_file = labels[os.path.basename(image_file)]
                gt_image = Image.open(gt_image_file)
                gt_image = gt_image.resize((image_shape[0], image_shape[1]))
                gt_image = np.asarray(gt_image)

                #create background image
                bkgnd_image = 255*np.ones((image_shape[0],image_shape[1]))
                bkgnd_image  = bkgnd_image - gt_image[:,:,0]
                bkgnd_image  = bkgnd_image - gt_image[:,:,1]
                bkgnd_image  = bkgnd_image - gt_image[:,:,2]
                gt_image = np.dstack((gt_image,bkgnd_image))

                # Appending them to existing batch
                x_train.append(image)  # x_train = np.append(x_train, [image], axis=0)
                y_train.append(gt_image)
            # y_train = to_categorical(y_train, num_classes=len(classes))

            batch_images = np.array(x_train)
            batch_lables = np.array(y_train)
            # normalize image data (not the labels)
            batch_images = batch_images.astype('float32') / 255
            batch_lables = batch_lables.astype('float32') / 255

            yield (batch_images, batch_lables)

def show_statistics(data_folder, fineGrained=False, title="Input Data Statistics"):

    images = glob(os.path.join(data_folder, 'image', '*.png'))
    print("\n" + ("#" * 70) + "\n" + ("#" * 21) + title+ ("#" * 21) + "\n" + ("#" * 70) )
    print("total image number \t " + str(len(images)))

    label_list = []
    label_counter = []
    for i in range(0, len(images)):
        image_file = images[i]

        # read labels from image_file names
        path, img_name = os.path.split(image_file)
        fn, ext = img_name.split(".")
        names = fn.split("_")
        if fineGrained:
            currLabel = names[1] + "_" + names[2]
        else:
            currLabel = names[1] #000001_triangle_blue_000001

        if np.isin(currLabel, label_list):
            loc = label_list.index(currLabel)
            label_counter[loc] += 1  # remove the class unknown
        else:
            label_list.append(currLabel)
            label_counter.append(1)

    print("total class number \t " + str(len(label_list)))
    for i in range(0, len(label_list)):
        print ("class " + str(label_list[i]) + " \t " + str(label_counter[i]) + " images")

    print(("#" * 70))

def plot_sample_classification_results(data, truth, classes, predictions,test_acc):
    # show testing results
    fig= plt.figure(figsize=(10, 10))
    fig.suptitle('Sample Classification Results: Prediction (Truth) and Test Accuracy: %' + str(round(100*test_acc, 2)))

    img_nbr_toshow = 25
    if len(data)< img_nbr_toshow:
        img_nbr_toshow = len(data)

    subplot_nbr = int(np.sqrt(img_nbr_toshow))

    for i in range(img_nbr_toshow):
        plt.subplot(subplot_nbr, subplot_nbr, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        sampleID = random.randint(1,len(data)-1)
        plt.imshow(data[sampleID], cmap=plt.cm.binary)
        pred_label=classes[np.argmax(predictions[sampleID])]
        true_label=classes[np.argmax(truth[sampleID])]
        caption= str(pred_label) + " (" + str(true_label + ")")
        plt.xlabel(caption)
    plt.show()

def plot_sample_segmentation_results(data, truth, predictions,test_acc):

    fig= plt.figure(figsize=(10, 10))
    fig.suptitle('Sample Segmentation Results: Test Accuracy: %' + str(round(100*test_acc, 2)))

    img_nbr_toshow = 8
    columns = img_nbr_toshow
    rows = 3
    for i in range(1, columns + 1):
        sampleID = i-1

        fig.add_subplot(rows, columns, i)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(data[sampleID])
        plt.xlabel("input image " + str(sampleID))

        fig.add_subplot(rows, columns, columns+i)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        truth_3Dimg = convert_4Dimgbatch_2_3Dimgbatch(truth)
        plt.imshow(truth_3Dimg[sampleID], cmap=plt.cm.binary)
        plt.xlabel("truth image " + str(sampleID))

        fig.add_subplot(rows, columns, 2*columns+i)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        pred_3Dimg = convert_4Dimgbatch_2_3Dimgbatch(predictions)
        plt.imshow(pred_3Dimg[sampleID], cmap=plt.cm.binary)
        plt.xlabel("predicted image " + str(sampleID))

    plt.show()

def plot_sample_data_with_groundtruth(data, truth):

    # show testing results
    fig= plt.figure(figsize=(10, 10))
    fig.suptitle('Data Samples')

    img_nbr_toshow = 4

    for i in range(img_nbr_toshow):
        plt.subplot(2, img_nbr_toshow, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        sampleID = i
        plt.imshow(data[sampleID], cmap=plt.cm.binary)
        plt.xlabel("input image " + str(sampleID))

        plt.subplot(2, img_nbr_toshow, img_nbr_toshow+ i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        truth_3Dimg = convert_4Dimgbatch_2_3Dimgbatch(truth)
        plt.imshow(truth_3Dimg[sampleID], cmap=plt.cm.binary)
        plt.xlabel("truth image " + str(sampleID))
    plt.show()

def convert_4Dimgbatch_2_3Dimgbatch(inputImgBatch):

    #print('inputImgBatch', inputImgBatch.shape, inputImgBatch.dtype, inputImgBatch.min(), inputImgBatch.max())
    outputImgBatch = np.zeros((inputImgBatch.shape[0], inputImgBatch.shape[1], inputImgBatch.shape[2], 3), dtype=np.float32)

    # generate one channel labelled image as ground truth
    color_map = {0: np.array([1, 0, 0]),
                 1: np.array([0, 1, 0]),
                 2: np.array([0, 0, 1]),
                 3: np.array([0, 0, 0])}

    for f in range(0, len(outputImgBatch)):

        curr_pred = inputImgBatch[f, :, :, :]
        #print('curr_pred', curr_pred.shape, curr_pred.dtype, curr_pred.min(), curr_pred.max())

        maxImg =np.argmax(curr_pred, axis=2)
        #print('maxImg', maxImg.shape, maxImg.dtype, maxImg.min(), maxImg.max())

        color_img = np.zeros((inputImgBatch.shape[1], inputImgBatch.shape[2], 3), dtype=np.float32)
        for i in range(0, maxImg.shape[0]):
            for j in range(0, maxImg.shape[1]):
                color_img[i, j] = color_map[maxImg[i,j]]

        outputImgBatch[f] = color_img

    return outputImgBatch
