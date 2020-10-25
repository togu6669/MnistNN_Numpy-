# checked with https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/ working 
# array model for backprop based on  https://sudeepraja.github.io/Neural/

# other nets with numpy
# https://pylessons.com/
# https://medium.com/analytics-vidhya/deep-neural-networks-step-by-step-with-numpy-library-565836a867db
# https://towardsdatascience.com/artificial-neural-network-implementation-using-numpy-and-classification-of-the-fruits360-image-3c56affa4491
# https://stackoverflow.com/questions/49007454/prepare-images-for-a-neural-network-model

# MNIST resources
# https://www.python-course.eu/neural_network_mnist.php

# discussion on the bias node and why we do not need it in MNIST
# http://makeyourownneuralnetwork.blogspot.com/2016/06/bias-nodes-in-neural-networks.html

import numpy as np
import NeuronLayers2
import LossFunc as lf
import ActFunc as af
import MNISTReader as mr
import pickle as pk
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# Read training images
#mr.ReadMNISTImages ('data/t10k-images-idx3-ubyte.gz')
images, image_size, image_count = mr.ReadMNISTImages ('data/train-images-idx3-ubyte.gz')
labels, label_count = mr.readMNISTLabels ('data/train-labels-idx1-ubyte.gz')

assert (image_count == label_count), "Number of training images is different then number of training labels!"

test_images, test_image_size, test_image_count = mr.ReadMNISTImages ('data/t10k-images-idx3-ubyte.gz')
test_labels, test_label_count = mr.readMNISTLabels ('data/t10k-labels-idx1-ubyte.gz')

assert (image_count == label_count), "Number of training images is different then number of training labels!"


fac = 0.99 / 255
nimages = np.asfarray (images) * fac + 0.01 # normalize grayscales to 0.01 - 1 https://www.python-course.eu/neural_network_mnist.php


# NUMPY 
# initialize network layer: No of Neurons, Previous Layer, Bias, Learning Rate, Activation Function
InputLayer = NeuronLayers2.NeuronFCLayer (image_size*image_size, None, 0, 0.5, af.PassThrough())
HiddenLayer1 = NeuronLayers2.NeuronFCLayer (128, InputLayer, 0.35, 0.5, af.ReLU()) # af.Sigmoid
OutputLayer = NeuronLayers2.NeuronFCLayer (10, HiddenLayer1, 0.6, 0.5, af.SoftMax(), lf.CrossEntropy()) # af.SoftMax 

InputLayer.setNextLayer (HiddenLayer1)
HiddenLayer1.setNextLayer (OutputLayer)
OutputLayer.setNextLayer(None)

start = timer()

No_of_epoch = 10
epoch = 0
accuracies = np.zeros(No_of_epoch)
# forthOfImages = images.shape [0] / 4

# plt.plot(minweight)

while epoch < No_of_epoch:
    img_count = 0
    success_count = 0
    # plt.axis([0, images.shape [0], -0.5, 0.5])
    plt.title ('Outputs H1 - red, O - blue')

    while img_count < images.shape [0]:
        # forward step
        # print ('--------------------- One epoche forward-----------------')
        # img = np.asfarray (images [img_count]) * fac + 0.01 # normalize grayscales to 0.01 - 1 https://www.python-course.eu/neural_network_mnist.php
        img = nimages[img_count].reshape (-1) # flaten to one dimension https://stackoverflow.com/questions/49007454/prepare-images-for-a-neural-network-model
        InputLayer.forward (img) #
        # print ('----------------------- Hidden Layer1 -------------------')
        HiddenLayer1.forward (None)
        # print ('----------------------- Output Layer -------------------')
        OutputLayer.forward (None)
        OutputLayer.lf.val (OutputLayer.Output(), labels [img_count])

        # backprop step 
        # print ('----------------------------------------------------------')
        # print ('--------------------- One epoche backprop-----------------')

        # label = labels [0]
        # if img_count < label_count:    # size of labels must be the same as size of images
        #     label = labels [img_count]
        
        OutputLayer.backward (labels [img_count]) # first layer
        HiddenLayer1.backward (None)

        OutputLayer.update()
        HiddenLayer1.update()

        if np.mod (img_count, 1000) == 0:
            print ("learning image #: ", img_count, " Label:", np.argmax (labels [img_count]), " Output:", np.argmax (OutputLayer.y))
            # plt.scatter (img_count, np.mean (OutputLayer.y), Color = 'blue')    
            # plt.scatter (img_count, np.mean (HiddenLayer1.y), Color = 'red')   
            # plt.pause (0.05)

        img_count = img_count + 1

        img_label = np.argmax (labels [img_count-1])
        net_output = np.argmax (OutputLayer.y)

        if (img_label == net_output):
            success_count = success_count + 1

    print ("epoch: ", epoch, " Label: ", img_label," Output: ", net_output, " Accurracy: ", success_count / img_count)
    accuracies [epoch] = success_count / img_count

    epoch = epoch + 1
   # plt.show ()

plt.plot(accuracies)
plt.xlabel('x - epoch')
plt.ylabel('y - training accuracies')
plt.title('Accuracies')
plt.show()


end = timer()
sectime = end - start
mintime = sectime / 60

print(' Time of training : ', sectime, ' sec, ', mintime, ' min')

def test(t_images, t_labels):
    img_count = 0
    success_count = 0
    test_img_count = t_images.shape [0]
    while img_count < test_img_count:

        # img = np.asarray(test_images [img_count]).squeeze()
        # plt.imshow(img)
        # plt.show()
    
        img = np.asfarray (t_images [img_count]) * fac + 0.01 # normalize grayscales to 0.01 - 1 https://www.python-course.eu/neural_network_mnist.php
        img = img.reshape (-1) # flaten to one dimension https://stackoverflow.com/questions/49007454/prepare-images-for-a-neural-network-model
        
        InputLayer.forward (img) 
        HiddenLayer1.forward(None)
        OutputLayer.forward(None)
        
        test_img_label = np.argmax (t_labels [img_count])
        net_output = np.argmax (OutputLayer.y)

        if (test_img_label == net_output):
            success_count = success_count + 1
    
        img_count = img_count + 1

    
    return success_count / test_img_count

print (" Accuracy : ", test (test_images, test_labels))

print ("------------------END OF MNIST LEARNING------------------")    
