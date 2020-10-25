# checked with https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/ working 
# array model for backprop based on  https://sudeepraja.github.io/Neural/

# MNIST data written digits database http://yann.lecun.com/exdb/mnist/

import numpy as np
import NeuronLayers
import LossFunc
import ActFunc as ne
import gzip 

InputLayer = NeuronLayers.NeuronFCLayer (2, None, None, 0, 0.5, ne.PassThroughAct)

# HiddenLayer = NeuronLayers.NeuronFCLayer (2, InputLayer, [[0.15, 0.20], [0.25, 0.30]], 0.35, 0.5, ne.SigmoidAct) 
# OutputLayer = NeuronLayers.NeuronFCLayer (2, HiddenLayer, [[0.4, 0.45], [0.5, 0.55]], 0.6, 0.5, ne.SigmoidAct) 

HiddenLayer1 = NeuronLayers.NeuronFCLayer (4, InputLayer, [[0.15, 0.20], [0.25, 0.30], [0.23, 0.77], [0.54, 0.11]], 0.35, 0.5, ne.SigmoidAct) 
HiddenLayer2 = NeuronLayers.NeuronFCLayer (3, HiddenLayer1, [[0.15, 0.20, 0.25, 0.30], [0.23, 0.77, 0.54, 0.11], [0.87, 0.12, 0.3, 0.21]], 0.35, 0.5, ne.SigmoidAct) 
OutputLayer = NeuronLayers.NeuronFCLayer (2, HiddenLayer2, [[0.40, 0.45, 0.5], [0.55, 0.6, 0.65]], 0.6, 0.5, ne.SigmoidAct) 

epoch = 0
while epoch < 2000:
    # forward test 
    # print ('--------------------- One epoche forward-----------------')
    InputLayer.forward([0.05, 0.1]) #
    # print ('----------------------- Hidden Layer1 -------------------')
    HiddenLayer1.forward(None)
    # print ('----------------------- Hidden Layer2 -------------------')
    HiddenLayer2.forward(None)
    # print ('----------------------- Output Layer -------------------')
    OutputLayer.forward(None)


    # backward test 
    # print ('----------------------------------------------------------')
    # print ('--------------------- One epoche backward-----------------')

    OutputLayer.backward (None, None, [0.01, 0.99]) # first layer
    HiddenLayer2.backward (OutputLayer.layerweights, OutputLayer.layerdelta, None)
    HiddenLayer1.backward (HiddenLayer2.layerweights, HiddenLayer2.layerdelta, None)

    OutputLayer.update()
    HiddenLayer2.update()
    HiddenLayer1.update()

    # print ("epoch: ", epoch, " Output:", OutputLayer.layeroutputs)
    # print (HiddenLayer.layeroutputs)
    epoch = epoch + 1

InputLayer.forward([0.2, 0.7])
HiddenLayer1.forward(None)
HiddenLayer2.forward(None)
OutputLayer.forward(None)


print ("Test1:", OutputLayer.layeroutputs)


InputLayer.forward([0.05, 0.1])
HiddenLayer1.forward(None)
HiddenLayer2.forward(None)
OutputLayer.forward(None)


print ("Test2:", OutputLayer.layeroutputs)
