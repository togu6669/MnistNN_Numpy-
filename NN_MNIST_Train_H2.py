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
import LossFunc
import ActFunc as ne
import MNISTReader as mr
from joblib import dump, load
import pickle as pk
from NN_Utils import SaveTheNet

# Read training images
#mr.ReadMNISTImages ('data/t10k-images-idx3-ubyte.gz')
images, image_size, image_count = mr.ReadMNISTImages ('data/train-images-idx3-ubyte.gz')
labels, label_count = mr.readMNISTLabels ('data/train-labels-idx1-ubyte.gz')

assert (image_count == label_count), "Number of training images is different then number of training labels!"

test_images, test_image_size, test_image_count = mr.ReadMNISTImages ('data/t10k-images-idx3-ubyte.gz')
test_labels, test_label_count = mr.readMNISTLabels ('data/t10k-labels-idx1-ubyte.gz')

assert (image_count == label_count), "Number of training images is different then number of training labels!"

# initialize network layer: No of Neurons, Previous Layer, Bias, Learning Rate, Activation Function
InputLayer = NeuronLayers2.NeuronFCLayer (image_size*image_size, None, 0, 0.5, ne.PassThroughAct)
HiddenLayer1 = NeuronLayers2.NeuronFCLayer (500, InputLayer, 0.35, 0.5, ne.SigmoidAct) 
HiddenLayer2 = NeuronLayers2.NeuronFCLayer (400, HiddenLayer1, 0.35, 0.5, ne.SigmoidAct) 
OutputLayer = NeuronLayers2.NeuronFCLayer (10, HiddenLayer2, 0.6, 0.5, ne.SigmoidAct) 
  
No_of_epoch = 5
epoch = 0
fac = 0.99 / 255
# forthOfImages = images.shape [0] / 4
while epoch < No_of_epoch:
    img_count = 0
    while img_count < images.shape [0]:
        # forward step
        # print ('--------------------- One epoche forward-----------------')
        img = np.asfarray (images [img_count]) * fac + 0.01 # normalize grayscales to 0.01 - 1 https://www.python-course.eu/neural_network_mnist.php
        img = img.reshape (-1) # flaten to one dimension https://stackoverflow.com/questions/49007454/prepare-images-for-a-neural-network-model
        InputLayer.forward (img) #
        # print ('----------------------- Hidden Layer1 -------------------')
        HiddenLayer1.forward (None)
        # # print ('----------------------- Hidden Layer2 -------------------')
        HiddenLayer2.forward (None)
        # print ('----------------------- Output Layer -------------------')
        OutputLayer.forward (None)

        # backprop step 
        # print ('----------------------------------------------------------')
        # print ('--------------------- One epoche backprop-----------------')

        label = labels [0]
        if img_count < label_count:    # size of labels must be the same as size of images
            label = labels [img_count]
        
        if np.mod (img_count, 1000) == 0:
            print ("learning image #: ", img_count, " Label:", labels [img_count], " Output:", OutputLayer.layeroutputs)

        OutputLayer.backward (None, None, labels [img_count]) # first layer
        HiddenLayer2.backward (OutputLayer.layerweights, OutputLayer.layerdelta, None)
        HiddenLayer1.backward (HiddenLayer2.layerweights, HiddenLayer2.layerdelta, None)
  
        OutputLayer.update()
        HiddenLayer2.update()
        HiddenLayer1.update()

        img_count = img_count + 1

    print ("epoch: ", epoch, " Label:", labels [img_count-1]," Output:", OutputLayer.layeroutputs)
    
    epoch = epoch + 1

    img_count = 0
    test_img_count = test_images.shape [0]
    test_img_count = 10
    
    if np.mod (epoch, 5) == 0:
        while img_count < test_img_count:

            # img = np.asarray(test_images [img_count]).squeeze()
            # plt.imshow(img)
            # plt.show()
        
            img = np.asfarray (test_images [img_count]) * fac + 0.01 # normalize grayscales to 0.01 - 1 https://www.python-course.eu/neural_network_mnist.php
            img = img.reshape (-1) # flaten to one dimension https://stackoverflow.com/questions/49007454/prepare-images-for-a-neural-network-model
            
            InputLayer.forward (img) 
            HiddenLayer1.forward(None)
            HiddenLayer2.forward(None)
            OutputLayer.forward(None)
            
            line = "Img Number: " + str (img_count) + " | Test value: " + str(np.argmax (test_labels [img_count])) + " | Network Response: " + str(np.argmax (OutputLayer.layeroutputs)) + "\n"

            filename = "MNIST-test-" + str(epoch) + ".txt"
            file2write = open (filename,'w')
            file2write.write (line)
            file2write.close()
            
            # save the entire network for the given test image
            # SaveTheNet (HiddenLayer1, HiddenLayer2, OutputLayer, epoch, img_count)
            # SaveTheNet1Hidden (HiddenLayer1, OutputLayer, epoch, img_count)

            img_count = img_count + 1

        print ("Test saved")
    
        file2write = open ("MNIST-network-epoch"+str(epoch)+".mninet",'wb')
        pk.dump(InputLayer, file2write)
        pk.dump(HiddenLayer1, file2write)
        pk.dump(HiddenLayer2, file2write)
        pk.dump(OutputLayer, file2write)
        file2write.close()

        # joblib the results every 10-epoche https://scikit-learn.org/stable/modules/model_persistence.html
        # dump (InputLayer, 'IL.joblib')
        # dump (HiddenLayer1, 'HL1.joblib')
        # dump (HiddenLayer2, 'HL2.joblib')
        # dump (OutputLayer, 'OL.joblib')

print ("------------------END OF MNIST LEARNING------------------")    
