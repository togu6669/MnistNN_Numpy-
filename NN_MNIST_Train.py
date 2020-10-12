# checked with https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/ working 
# array model for backprop based on  https://sudeepraja.github.io/Neural/

# other nets with numpy
# https://pylessons.com/
# https://medium.com/analytics-vidhya/deep-neural-networks-step-by-step-with-numpy-library-565836a867db
# https://towardsdatascience.com/artificial-neural-network-implementation-using-numpy-and-classification-of-the-fruits360-image-3c56affa4491
# https://stackoverflow.com/questions/49007454/prepare-images-for-a-neural-network-model

# MNIST resources
# https://www.python-course.eu/neural_network_mnist.php

import numpy as np
import NeuronLayers2
import LossFunc as lf
import ActFunc as ne
import MNISTReader as mr
import pickle as pk
from NN_Utils import SaveTheNet
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
InputLayer = NeuronLayers2.NeuronFCLayer (image_size*image_size, None, 0, 0.5, ne.PassThrough)
HiddenLayer1 = NeuronLayers2.NeuronFCLayer (128, InputLayer, 0.35, 0.5, ne.ReLU) # ne.ReLU
OutputLayer = NeuronLayers2.NeuronFCLayer (10, HiddenLayer1, 0.6, 0.5, ne.SoftMax, lf.CrossEntropy) # ne.SoftMax 
# HiddenLayer1 = NeuronLayers2.NeuronFCLayer (500, InputLayer, 0.35, 0.5, ne.Sigmoid) 
# HiddenLayer2 = NeuronLayers2.NeuronFCLayer (400, HiddenLayer1, 0.35, 0.5, ne.Sigmoid) 
# OutputLayer = NeuronLayers2.NeuronFCLayer (10, HiddenLayer1, 0.6, 0.5, ne.Sigmoid) 


# TORCH 

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
        OutputLayer.LossFunc (OutputLayer.Output(), labels [img_count])

        # backprop step 
        # print ('----------------------------------------------------------')
        # print ('--------------------- One epoche backprop-----------------')

        # label = labels [0]
        # if img_count < label_count:    # size of labels must be the same as size of images
        #     label = labels [img_count]
        
        OutputLayer.backward (None, None, labels [img_count]) # first layer
        HiddenLayer1.backward (OutputLayer.layerweights, OutputLayer.layerdelta, None)

        OutputLayer.update()
        HiddenLayer1.update()

        if np.mod (img_count, 1000) == 0:
            print ("learning image #: ", img_count, " Label:", np.argmax (labels [img_count]), " Output:", np.argmax (OutputLayer.layeroutputs))
            plt.scatter (img_count, np.mean (OutputLayer.layeroutputs), Color = 'blue')    
            plt.scatter (img_count, np.mean (HiddenLayer1.layeroutputs), Color = 'red')   
            plt.pause (0.05)

        img_count = img_count + 1

        img_label = np.argmax (labels [img_count-1])
        net_output = np.argmax (OutputLayer.layeroutputs)

        if (img_label == net_output):
            success_count = success_count + 1

    print ("epoch: ", epoch, " Label: ", img_label," Output: ", net_output, " Accurracy: ", success_count / img_count)
    accuracies [epoch] = success_count / img_count

    epoch = epoch + 1
    plt.show ()

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
        net_output = np.argmax (OutputLayer.layeroutputs)

        if (test_img_label == net_output):
            success_count = success_count + 1
        
        # line = "Img Number: " + str (img_count) + " | Test value: " + str(test_img_label) + " | Network Response: " + str(net_output) + "\n"
        # filename = "MNIST-test-" + str(epoch) + ".txt"
        # file2write = open (filename,'w')
        # file2write.write (line)
        # file2write.close()
        
        # # save the entire network for the given test image
        # SaveTheNet (HiddenLayer1, HiddenLayer2, OutputLayer, epoch, img_count)
        # SaveTheNet1Hidden (HiddenLayer1, OutputLayer, epoch, img_count)

        img_count = img_count + 1

    # print ("Test saved")

    # file2write = open ("MNIST-network-epoch"+str(epoch)+".mninet",'wb')
    # pk.dump(InputLayer, file2write)
    # pk.dump(HiddenLayer1, file2write)
    # pk.dump(OutputLayer, file2write)
    # file2write.close()

    return success_count / test_img_count

print (" Accuracy : ", test (test_images, test_labels))

print ("------------------END OF MNIST LEARNING------------------")    
