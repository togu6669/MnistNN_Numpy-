import numpy as np
import math as mt
import ActFunc
# import LossFunc

class NeuronFCLayer:        # fully connected
    def __init__(self, NNeurons, prevLayer, weights, bias, learnRate, actFunc):
        self.LearnRate = learnRate
        self.ActFunc = actFunc
        self.Bias = bias

        self.layerdelta = None # shape
        self.layerinputs = None # shape
        if weights is not None:
            self.layerweights = np.array (weights)
        else:
            self.layerweights = np.zeros (NNeurons)
        self.layeroutputs = np.zeros (NNeurons) # shape
        self.pLayer = prevLayer         # connections to prevLayer

    def Output(self):
        a = self.layerinputs.reshape (self.layerinputs.size, -1)
        a = np.multiply (a, np.transpose (np.atleast_2d (self.layerweights)))
        a = self.ActFunc (np.sum (a, 0) + self.Bias)
        return a

    def forward(self, inputs):  # propagte signal from 1st to n-layer
        if inputs is not None:  # 1st input layer, in = out
            self.layerinputs = np.array(inputs)
            self.layeroutputs = self.layerinputs 
        elif self.pLayer is not None: 
            self.layerinputs = self.pLayer.layeroutputs
            self.layeroutputs = self.Output()              
        # print ('Inputs ', self.layerinputs)
        # print ('Weights ', self.layerweights)
        # print ('Output ', self.layeroutputs)

    def backward(self, nextlayerweights, nextlayerdelta, Y):
        # np.transpose will work only on 2d array/matrix thus np.atleast_2d

        if nextlayerdelta is None:
            dLoss_or_delta = self.layeroutputs - Y # loss function derivative
        else: 
            # sum of (transponed next laser weights * next layer deltas)
            dLoss_or_delta = np.dot (nextlayerweights.T, nextlayerdelta) 
        # activation function derivative
        dAct_dVal = self.layeroutputs * (1 - self.layeroutputs)
        # dAct * loss or transponed next layer weights * next layer deltas 
        self.layerdelta = dLoss_or_delta * dAct_dVal
        
        # detlas * transponed (reshaped) value derivative (input)
        dVal_dWeight = self.layerinputs
        self.weightdeltas = self.layerdelta * dVal_dWeight.reshape (dVal_dWeight.size, -1)
        
        # print ('Inputs ', self.layerinputs)
        # print ('Layer Deltas ', self.layerdelta)
        # print ('Deltas ', self.weightdeltas)
        # print ('Weights ', self.layerweights)

        return self.layerdelta, self.layerweights

    def update (self):
        self.layerweights = self.layerweights - self.LearnRate * self.weightdeltas.T
