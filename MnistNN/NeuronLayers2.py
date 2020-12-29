import numpy as np
import math as mt
import ActFunc as ac
from numpy.linalg import norm
# import LossFunc

import torch
import torch.nn.functional as F

# set numpy float format to 2 decimal places
# https://stackoverflow.com/questions/22222818/how-to-printing-numpy-array-with-3-decimal-places
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
 
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
        self.sum_xwT_b = None # layer x * wT + b
        self.y = np.zeros (self.nn) # output shape


    def Output(self):
        # x * wT + B is linear 
        x = self.x.reshape (self.x.size, -1)
        z = x * np.transpose (np.atleast_2d (self.w))
        self.z_b = np.sum (z, 0) + self.b 
        # activation func is not linear 
        y = self.af.val (self.z_b)
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

    # softmax + cross entropy derivative clearly explained
    # https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba
    # discusion on backprof with softmax + cross entropy (derivatives)
    # https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
    
    # good example on backprop with softmax and cross entropy
    # https://towardsdatascience.com/dismantling-neural-networks-to-understand-the-inner-workings-with-math-and-pytorch-beac8760b595

    # when use the dot and hadamard product (just before sigmoid explanation)
    # https://aew61.github.io/blog/artificial_neural_networks/1_background/1.b_activation_functions_and_derivatives.html

    def backward(self, Y):

        if self.nl is None:
            # torch_w_grad, _ = getTorchGrad()
            self.dl = self.lf.d_val (self.y, Y)
        else: 
            # next layer weights * next layer deltas
            self.dl = np.dot (self.nl.d, self.nl.w)
            
        self.dy = self.af.d_val (self.z_b) 

        # dz = dx / dw input derivative as Jacobian 
        # https://aimatters.wordpress.com/2020/06/14/derivative-of-softmax-layer/
        # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        # dz = np.zeros ((self.z_b.size, self.w.size))
        # for k in range(0, self.z_b.shape[0]):
        #     for i in range(0, self.w.shape[1]):
        #         for j in range(0, self.w.shape[0]):
        #             if i == k:
        #                 dz[k][(i*self.w.shape[0]) + j] = self.x[j]
        
        # dz = dx / dw as x, however it is important to remember that it is a Jacobian! 
        dz = self.x.reshape (1, self.x.shape [0])

        # dot or Hadamard? Make a choice BASED on the data shape, 
        if (self.dy.ndim > 1): 
            if (self.dy.ndim == 2):
                self.d = np.dot (self.dl, self.dy)  # vector * jacobian, jacobian dimension reduction!!!
        else:
            self.d = self.dl * self.dy  # Hadamard element-wise vector * vector

        # if we would have dw as a Jacobian
        # self.dw = np.dot (self.d, dz).reshape (self.w.shape)

        # reshape dz to (1, N) and d to (M, 1) to obtain dw of (M, N) shape
        self.dw = np.dot (self.d.reshape (self.d.shape[0], 1), dz)

        # difference between torch and math only for the last layer
        # if nextlayerdelta is None:
        #     c = torch.from_numpy (self.dw.T)
        #     dif = torch_w_grad - c
        #     self.dw = np.array (torch_w_grad).T

        # print ('Y  : ', self.y)
        # print ('L  : ', Y)
        # print ('W  : ', self.w)
        # print ('dW : ', self.dw)

    def update (self):
        self.w = self.w - self.lr * self.dw

        
    def outputLossFunc ():
        if self.lf is not None:
            a = self.lf (self.Output())
            print ('Loss Funcion value ', a)  

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
