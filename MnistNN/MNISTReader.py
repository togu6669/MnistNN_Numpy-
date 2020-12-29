# MNIST data written digits database http://yann.lecun.com/exdb/mnist/

# Reading and flatening the images the images 
# https://stackoverflow.com/questions/49007454/prepare-images-for-a-neural-network-model
# https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python

import numpy as np
import gzip as gz
# import matplotlib.pylab as plt
import os 


def ReadMNISTImages (FileName):
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, FileName)
    # print (filename)

    # isinstance (images, float)

    # with gz.open(filename, 'r') as fin:        
    #     images = fin.read()
        # print(type (images))

    fin =  gz.open(filename, 'r')        
    image_size = 28

    fin.read (4)
    img_count = fin.read (4)
    num_images = int.from_bytes(img_count, "big")
    fin.read(8)

    buf = fin.read (image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size)
    # data = data.reshape(num_images, image_size, image_size, 1)

    # image = np.asarray(data[0]).squeeze()
    # plt.imshow(image)
    # plt.show()
    
    # image = np.asarray(data[2]).squeeze()
    # plt.imshow(image)
    # plt.show()
    
    # image = np.asarray (data[num_images-1]).squeeze()
    # plt.imshow (image)
    # plt.show()
    
    return data, image_size, num_images

def readMNISTLabels (FileName):
    f = gz.open(FileName,'r')
    f.read(4)
    lables_count = int.from_bytes (f.read(4), "big")
    buf = f.read (lables_count)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.float)

    no_of_different_labels = 10
    lr = np.arange (no_of_different_labels)
    labels_one_hot = np.arange (lables_count*no_of_different_labels).reshape (lables_count, no_of_different_labels).astype(np.float)

    # transform labels into one hot representation
    for label in range(lables_count): 
        labels_one_hot[label] = (lr==labels[label]).astype(np.float)
        # we don't want zeroes and ones in the labels neither:
        a = labels_one_hot[label]
        a [a==0] = 0.01
        a [a==1] = 0.99
        labels_one_hot[label] = a
 
    # for i in range(0,50):   
    #     buf = f.read(1)
    #     labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    #     print(labels)
    return labels_one_hot, lables_count

labels = readMNISTLabels ('data/train-labels-idx1-ubyte.gz')
