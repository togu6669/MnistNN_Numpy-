import numpy as np
import math as mt
import ActFunc as ac
from numpy.linalg import norm
# import LossFunc

import torch
import torch.nn.functional as F

class NeuronFCLayer:        # fully connected
    def __init__(self, nneurons, prevLayer, bias, learnRate, actFunc, lossFunc = None):
        self.LearnRate = learnRate
        self.ActFunc = actFunc
        self.LossFunc = lossFunc
        self.Bias = bias
        self.NNeurons = nneurons

        self.layerdelta = None # shape
        self.layerinputs = None # shape
        if prevLayer is not None:
            # https://www.python-course.eu/neural_networks_with_python_numpy.php
            weights = np.random.uniform(-1., 1., self.NNeurons*prevLayer.NNeurons).reshape (self.NNeurons, prevLayer.NNeurons)
            # the above seems to be not better then rand
            # weights = np.random.rand (self.NNeurons, prevLayer.NNeurons) # uniformly random weights  current NNeurons rows x prevLayer.NNeurson columns
            # normalizing the weights vector to fixed L2 https://machinelearningmastery.com/vector-norms-machine-learning/
            a = norm (weights) # sqrt (sum (sqr(weight1)..sqr(weightn)))
            weights = weights / a 
            # other approaches https://stackoverflow.com/questions/40816269/training-mnist-dataset-with-sigmoid-activation-function
            # a = mt.sqrt (6./(prevLayer.NNeurons + self.NNeurons)) # glorot initialization 
            # weights = weights * a 
            self.layerweights = np.array (weights)
        else:
            self.layerweights = np.zeros (self.NNeurons) # input layer
        self.layeroutputs = np.zeros (self.NNeurons) # shape
        self.pLayer = prevLayer         # connections to prevLayer

    def Output(self):
        #  W x X + B is linear 
        a = self.layerinputs.reshape (self.layerinputs.size, -1)
        a = np.multiply (a, np.transpose (np.atleast_2d (self.layerweights)))
        # activation func is not linear 
        a = self.ActFunc (a, self.Bias)
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

    def outputLossFunc ():
        if self.LossFunc is not None:
            a = self.LossFunc (self.Output())
            print ('Loss Funcion value ', a)  

    # softmax + cross entropy derivative clearly explained
    # https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba
    # discusion on backprof with softmax + cross entropy (derivatives)
    # https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
    
    # good example on backprop with softmax and cross entropy
    # https://towardsdatascience.com/dismantling-neural-networks-to-understand-the-inner-workings-with-math-and-pytorch-beac8760b595

    def backward(self, nextlayerweights, nextlayerdelta, Y):

        # torch autograd for comparison with my math
        def getTorchGrad ():
            w = torch.from_numpy (self.layerweights)
            w = torch.tensor (w, requires_grad = True)
            b = torch.tensor (self.Bias, requires_grad = True)
            b = torch.reshape (b, (1, 1))
            x = torch.from_numpy (self.layerinputs)
            x = torch.reshape (x, (1, list (x.size())[0] ))
            z = x @ w.T + b
            
            # out = torch.from_numpy (self.layeroutputs)
            # out = torch.reshape (out, (1, list (out.size())[0] ))
            label = np.array (np.argmax(Y)).reshape (1)
            label = torch.tensor (label) 
            
            # torch_ce_out = F.cross_entropy(out, label)
            torch_ce = F.cross_entropy(z, label)
            
            torch_ce.backward() # torch autograd
            # torch_w_grad = w.grad
            # torch_b_grad = b.grad
            return w.grad, b.grad


        # np.transpose will work only on 2d array/matrix thus np.atleast_2d - not used anymore
        if nextlayerdelta is None:
            torch_w_grad, _ = getTorchGrad()
            
            
            dLoss_or_delta = self.layeroutputs - Y # loss function derivative
            
            # activation function derivative
            if self.ActFunc is ac.Sigmoid:
                dAct_dVal = self.layeroutputs * (1 - self.layeroutputs)
            # https://stackoverflow.com/questions/45648668/convert-numpy-array-to-0-or-1
            elif self.ActFunc is ac.ReLU:
                dAct_dVal = np.where (self.layeroutputs < 0, 0, 1)
            elif self.ActFunc is ac.SoftMax:            
                dAct_dVal = 1 
        else: 
            # sum of (transponed next laser weights * next layer deltas)
            dLoss_or_delta = np.dot (nextlayerweights.T, nextlayerdelta) 
            
            # activation function derivative
            if self.ActFunc is ac.Sigmoid:
                dAct_dVal = self.layeroutputs * (1 - self.layeroutputs)
            elif self.ActFunc is ac.ReLU:
                dAct_dVal = np.where (self.layeroutputs < 0, 0, 1)
            elif self.ActFunc is ac.SoftMax:            
                dAct_dVal = 1

        # dAct * loss or transponed next layer weights * next layer deltas 
        self.layerdelta = dLoss_or_delta * dAct_dVal
        
        # detlas * transponed (reshaped) input derivative: d(Val*Weight)
        dInput = self.layerinputs
        self.weightdeltas = self.layerdelta * dInput.reshape (dInput.size, -1)
        
        # difference between torch and math only for the last layer
        if nextlayerdelta is None:
            c = torch.from_numpy (self.weightdeltas.T)
            dif = torch_w_grad - c
            self.weightdeltas = np.array (torch_w_grad).T


        # print ('Inputs ', self.layerinputs)
        # print ('Layer Deltas ', self.layerdelta)
        # print ('Deltas ', self.weightdeltas)
        # print ('Weights ', self.layerweights)

        return self.layerdelta, self.layerweights

    def update (self):
        self.layerweights = self.layerweights - self.LearnRate * self.weightdeltas.T




    # working version for Sigmoid Activation and squarred error loss function (sum (target - output)^2)
    def backwardSigmoid (self, nextlayerweights, nextlayerdelta, Y):
        # np.transpose will work only on 2d array/matrix thus np.atleast_2d - not used anymore

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

