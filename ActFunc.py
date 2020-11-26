# very simply put "in out" explanation 
# https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c

import numpy as np
import math as mt
import torch
import torch.nn.functional as F
import abc

class ActFunc (metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def val(self, y):
        pass

    # @abc.abstractmethod
    # def val_torch(self, y):
    #     pass       mmmmm mm m m,m üpßvv                        yn<ilwuh<98p8t0ußuz                                                                                                                             oäji9i4    ßu9ßßßüi09iiü´ßß9ßdf    9   00zußp

    @abc.abstractmethod
    def d_val (self, y):   
        pass

class ReLU(ActFunc):
    # def __init__(self): , öl,.,.           bn bb , , o8896ühjpö
    #     _

    def val (self, y):
        # https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
        # there are at least 3 ways to max the array elements 
        # 1. np.maximum(y, 0) medium fast 
        # 2. y * (y > 0) fastest 
        # 3. (abs (y) + y) / 2 slowest 
        # assert (np.max (y) <= 1), "bigger then 1"
        b = y * (y > 0) 
        np.clip (b, 0.01, 0.99, b) # without clipping relu values are > 1 and it seems to cause that it never converge
        return b

    # def val_torch (self, y):
        # b = torch.from_numpy (y)
        # b = F.relu(b)
        # b = torch.clamp(b, 0.01, 0.99)
        # return b

    def d_val (self, y):
        # https://stackoverflow.com/questions/42042561/relu-derivative-in-backpropagation
        # the derivative is 0 for x < 0, and 1 for x >= 0
        return np.where (y < 0, 0, 1)

# sigmoid is not y-zero centric (by x=0 it y = 0.5), and its gradient vanishes to zero when x -> -inf, +inf
class Sigmoid(ActFunc):

    def val (self, y):
        assert np.isfinite (1 / (1 + np.exp (-y))).all()
        return 1 / (1 + np.exp (-y))

    def d_val (self, y):
        return self.val (y) * (1 - self.val (y))


class SoftPlus(ActFunc): # this has been not checked

    def val (self, y):
        return np.log (1 + np.exp (y))

    def d_val (self, y):
        a = np.exp (y) 
        return a / (1 + a)

# softmax discussion 
# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
# https://cs231n.github.io/linear-classify/#softmax

# https://github.com/pangolulu/neural-network-from-scratch 
# ReLU vs sigmoid in mnist example
# https://datascience.stackexchange.com/questions/18667/relu-vs-sigmoid-in-mnist-example - cliping softmax

# Softmax derivative - jacobian discussion
# https://stackoverflow.com/questions/36279904/softmax-derivative-in-numpy-approaches-0-implementation
class SoftMax(ActFunc):

    def val (self, y):
        y = y - np.max (y)
        b = np.exp (y)
        b = b / np.sum (b, 0)
        np.clip (b, 0.01, 0.99, b) # clip the softmax result to comply with true output values in a loss function
        # assert np.isfinite(a).all()
        # assert np.isfinite(b).all()
        return b

    # def val_torch (self, y):
        # b = torch.from_numpy (y)
        # b = F.softmax(b)
        # b = torch.clamp(b, 0.01, 0.99)
        # return b

    
    # taken from https://aerinykim.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
    def d_val (self, y):
        s = self.val(y).reshape(-1,1)
        s1 = np.diagflat(s) - np.dot(s, s.T)
        return s1
        
    # taken from https://towardsdatascience.com/dismantling-neural-networks-to-understand-the-inner-workings-with-math-and-pytorch-beac8760b595
    # while the sigmoid function is the function of one neuron, the softmax is a multivariate function of many neurons 
        # sm = self.val(y).squeeze()
        # sm_size = sm.shape[0]
        # sm_ps = []
        # for i, sm_i in enumerate(sm):
        #     for j, sm_j in enumerate(sm):
        #     # First case: i and j are equal:
        #         if(i==j):
        #             # Differentiating the softmax of a neuron w.r.t to itself
        #             sm_p = sm_i * (1 - sm_i)
        #             sm_ps.append(sm_p)
        #     # Second case: i and j are not equal:
        #         else:
        #             # Differentiating the softmax of a neuron w.r.t to another neuron
        #             sm_p = -sm_i * sm_j
        #             sm_ps.append(sm_p)
        # sm_ps = np.array(sm_ps).reshape (sm_size, sm_size)
        # return sm_ps       

class PassThrough(ActFunc):

    def val (self, y):
        return y

    def d_val (self, y):
        return y

# hyperbolic tangen function y = (2 / (1+e^-2x)) + 1,  is zero centric

