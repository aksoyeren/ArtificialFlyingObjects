import re
import random
import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy.misc
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
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

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
            batch_images = batch_images.astype('float32') / 255

            yield (batch_images, batch_lables)

def generate_augmented_classification_batches(in_gen, image_gen):
    for data, labels in in_gen:
        aug_data = image_gen.flow(255 * data, labels, batch_size=data.shape[0])

        aug_img, aug_lab = next(aug_data)

        yield aug_img / 255.0, aug_lab
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

