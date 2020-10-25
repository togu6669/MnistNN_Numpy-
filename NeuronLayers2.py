import numpy as np
import math as mt
import ActFunc as ac
from numpy.linalg import norm
# import LossFunc

import torch
import torch.nn.functional as F

class NeuronFCLayer:        # fully connected
    def __init__(self, nneurons, prevLayer, bias, learnRate, actFunc, lossFunc = None):
        self.lr = learnRate
        self.af = actFunc
        self.lf = lossFunc
        self.b  = bias
        self.nn = nneurons

        self.d = None # deltas shape
        self.x = None # input shape
        self.pl = prevLayer         # connections to prevLayer
        if self.pl is not None:
            # https://www.python-course.eu/neural_networks_with_python_numpy.php
            w = np.random.uniform(-1., 1., self.nn*self.pl.nn).reshape (self.nn, self.pl.nn)
            # the above seems to be not better then rand
            # w = np.random.rand (self.nn, prevLayer.nn) # uniformly random weights  current nn rows x prevLayer.nn columns
            # normalizing the weights vector to fixed L2 https://machinelearningmastery.com/vector-norms-machine-learning/
            a = norm (w) # sqrt (sum (sqr(weight1)..sqr(weightn)))
            w = w / a 
            # other approaches https://stackoverflow.com/questions/40816269/training-mnist-dataset-with-sigmoid-activation-function
            # a = mt.sqrt (6./(prevLayer.nn + self.nn)) # glorot initialization 
            # w = w * a 
            self.w = np.array (w)
        else:
            self.w = np.zeros (self.nn) # input layer
        self.y = np.zeros (self.nn) # output shape

    def Output(self):
        # x * wT + B is linear 
        x = self.x.reshape (self.x.size, -1)
        xwT = np.multiply (x, np.transpose (np.atleast_2d (self.w)))
        sum_xwT_b = np.sum (xwT, 0) + self.b
        # activation func is not linear 
        y = self.af.val (sum_xwT_b)
        return y

    def forward(self, inputs):  # propagte signal from 1st to n-layer
        if inputs is not None:  # 1st input layer, in = out
            self.x = np.array(inputs)
            self.y = self.x 
        elif self.pl is not None: 
            self.x = self.pl.y
            self.y = self.Output()              
        # print ('Inputs ', self.x)
        # print ('Weights ', self.w)
        # print ('Output ', self.y)
    
    def setNextLayer (self, nextLayer):
        self.nl = nextLayer

    def outputLossFunc ():
        if self.lf is not None:
            a = self.lf (self.Output())
            print ('Loss Funcion value ', a)  

    # softmax + cross entropy derivative clearly explained
    # https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba
    # discusion on backprof with softmax + cross entropy (derivatives)
    # https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
    
    # good example on backprop with softmax and cross entropy
    # https://towardsdatascience.com/dismantling-neural-networks-to-understand-the-inner-workings-with-math-and-pytorch-beac8760b595

    def backward(self, Y):

        # torch autograd for comparison with my math
        def getTorchGrad ():
            w = torch.from_numpy (self.w)
            w = torch.tensor (w, requires_grad = True)
            b = torch.tensor (self.b, requires_grad = True)
            b = torch.reshape (b, (1, 1))
            x = torch.from_numpy (self.x)
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
        if self.nl is None:
            # torch_w_grad, _ = getTorchGrad()
            self.dl = self.lf.d_val (self.y, Y)
        else: 
            # sum of (transponed next laser weights * next layer deltas)
            self.dl = np.dot (self.nl.w.T, self.nl.dl*self.nl.dy) 
            
        self.dy = self.af.d_val(self.y)

        #derivative of loss function * derivative of activation funtion  * transponed (reshaped) input derivative
        # input derivative dx / dw = x aka self.x
       
        self.dw = self.dl * self.dy * self.x.reshape (self.x.size, -1)
        
        # difference between torch and math only for the last layer
        # if nextlayerdelta is None:
        #     c = torch.from_numpy (self.dw.T)
        #     dif = torch_w_grad - c
        #     self.dw = np.array (torch_w_grad).T

        # print ('Inputs ', self.x)
        # print ('Deltas ', self.dw)
        # print ('Weights ', self.w)

    def update (self):
        self.w = self.w - self.lr * self.dw.T