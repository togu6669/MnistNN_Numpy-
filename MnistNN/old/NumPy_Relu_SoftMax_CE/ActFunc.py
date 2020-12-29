import numpy as np
import math as mt
import torch
import torch.nn.functional as F

def ReLU(value, bias):
    a = np.sum(value, 0) + bias
    # https://stackoverflow.com/questions/42042561/relu-derivative-in-backpropagation
    # the derivative is 0 for x < 0, and 1 for x >= 0
    # https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
    # there are at least 3 ways to max the array elements 
    # 1. np.maximum(a, 0) medium fast 
    # 2. a * (a > 0) fastest 
    # 3. (abs (a) + a) / 2 slowest 
    # assert (np.max (a) <= 1), "bigger then 1"
    b = a * (a > 0) 
    np.clip (b, 0.01, 0.99, b) # without clipping relu values are > 1 and it seems to cause that it never converge
    
    # torch_relu = torch.from_numpy (a)
    # torch_relu = F.relu(torch_relu)
    # torch_relu = torch.clamp(torch_relu, 0.01, 0.99)
    # c = torch.from_numpy (b) - torch_relu
    return b

# sigmoid is not y-zero centric (by x=0 it y = 0.5), and its gradient vanishes to zero or fast goes to infinity
def Sigmoid(value, bias):
    a = np.sum(value, 0) + bias
    assert np.isfinite (1 / (1 + np.exp (-a))).all()
    return 1 / (1 + np.exp (-a))

def SoftPlus(value, bias): # this has been not checked
    a = np.sum(value, 0) + bias
    return np.log (1 + np.exp (a))

# softmax discussion 
# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
# https://cs231n.github.io/linear-classify/#softmax

# https://github.com/pangolulu/neural-network-from-scratch 
# ReLU vs sigmoid in mnist example
# https://datascience.stackexchange.com/questions/18667/relu-vs-sigmoid-in-mnist-example - cliping softmax

def SoftMax(value, bias):
    a = np.sum(value, 0) + bias
    a = a - np.max (a)
    b = np.exp(a) / np.sum(np.exp(a), 0)
    np.clip (b, 0.01, 0.99, b) # clip the softmax result to comply with true output values in a loss function
    # assert np.isfinite(a).all()
    # assert np.isfinite(b).all()
    # z = torch.from_numpy (a)
    # c = F.softmax(z)
    # c = torch.clamp(c, 0.01, 0.99)
    # c = torch.from_numpy (b) - c
    return b

def PassThrough (value, bias):
    return value

# hyperbolic tangen function y = (2 / (1+e^-2x)) + 1,  is zero centric

