import numpy as np
import NeuronLayers2
import LossFunc
import ActFunc as ne
import MNISTReader as mr
# from joblib import dump, load
# import matplotlib.pylab as plt
import pickle as pk
from timeit import default_timer as timer


images, image_size, image_count = mr.ReadMNISTImages ('data/t10k-images-idx3-ubyte.gz')
labels, label_count = mr.readMNISTLabels ('data/t10k-labels-idx1-ubyte.gz')

# images, image_size, image_count = mr.ReadMNISTImages ('data/train-images-idx3-ubyte.gz')
# labels, label_count = mr.readMNISTLabels ('data/train-labels-idx1-ubyte.gz')

assert (image_count == label_count), "Number of training images is different then number of training labels!"

#file2write = open ("29-02-20 56 epochs\MNIST-network-epoch"+str(55)+".mninet",'rb')
file2write = open ("MNIST-network-epoch"+str(5)+".mninet",'rb')
# file2write = open ("MNIST.net",'rb')
InputLayer = pk.load (file2write)
HiddenLayer1 = pk.load (file2write)
HiddenLayer1.pLayer = InputLayer
HiddenLayer2 = pk.load (file2write)
HiddenLayer2.pLayer = HiddenLayer1
OutputLayer = pk.load (file2write)
OutputLayer.pLayer = HiddenLayer2
file2write.close()

# file2write = open ("MNIST-network-epoch"+str(0)+".mninet",'rb')
# InputLayer0 = pk.load (file2write)
# HiddenLayer10 = pk.load (file2write)
# HiddenLayer20 = pk.load (file2write)
# OutputLayer0 = pk.load (file2write)
# file2write.close()

# load netwrok from joblib serialized files https://scikit-learn.org/stable/modules/model_persistence.html
# InputLayer = load('IL.joblib')  
# HiddenLayer1 = load ('HL1.joblib')
# HiddenLayer2 = load ('HL2.joblib')
# OutputLayer = load ('OL.joblib')

fac = 0.99 / 255

img_count = 0
errorrate = 0.0

start = timer()

while img_count < images.shape [0]:

    # img = np.asarray(images [img_count]).squeeze()
    # plt.imshow(img)
    # plt.show()

    img = np.asfarray (images [img_count]) * fac + 0.01 # normalize grayscales to 0.01 - 1 https://www.python-course.eu/neural_network_mnist.php
    img = img.reshape (-1) # flaten to one dimension https://stackoverflow.com/questions/49007454/prepare-images-for-a-neural-network-model
    
    InputLayer.forward (img) 
    HiddenLayer1.forward(None)
    HiddenLayer2.forward(None)
    OutputLayer.forward(None)
    
    # InputLayer0.forward (img) 
    # HiddenLayer10.forward(None)
    # HiddenLayer20.forward(None)
    # OutputLayer0.forward(None)

    # od = OutputLayer.layeroutputs - OutputLayer0.layeroutputs
    
    # line = "Img Number: " + str (img_count) + " | Test value: " + str(np.argmax (labels [img_count])) + " | Network Response: " + str(np.argmax (OutputLayer.layeroutputs)) + "\n"
    # print (line)
    if np.argmax (OutputLayer.layeroutputs) != np.argmax (labels [img_count]): 
        errorrate = errorrate + 1.0

    img_count = img_count + 1
    
print("error rate: " + str(errorrate/img_count))

end = timer()
sectime = end - start
mintime = sectime / 60
print(' Time of training : ', sectime, ' sec, ', mintime, ' min')


