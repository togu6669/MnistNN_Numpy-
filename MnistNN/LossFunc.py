# discussions of different loss functions
# https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/
# https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

import numpy as np
import matplotlib.pyplot as plt
import abc

class LossFunc (metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def val(self, y, l):
        pass

    # @abc.abstractmethod
    # def val_torch(self, y):
    #     pass

    @abc.abstractmethod
    def d_val (self, y, l):   
        pass

class CrossEntropy(LossFunc):

    def val (self, y, l):
        # a = - np.log (Output)        # a = Label * a
        # a = np.sum (a)
        # a = np.zeros (l)
        # i = np.where(l == 0.99)
        # a [i[0]] = 1
        return - np.sum ((l*np.log (y)), 0)

    def d_val (self, y, l):
        z = np.zeros_like(y)
        a = np.where(l == 0.99) # awful hack, we know that the value under the correct label index equals 0.99
        if len(a) > 0 and len(a[0]) > 0:
            z[a[0]] = -1 / y [a[0]]
        return z 

class MeanSquareError(LossFunc):

    def val (self, y, l):
        n = y.size 
        return np.sum (np.power(l-y, 2)) / (2 * n)
        # return np.sum (np.power(l-y, 2)) / 2    # 2 classes 

    # the derivative is in fact - np.sum (l-y) / n, where l - label, y- the network output and n is the number of neurons in the output layer 
    # but here we have one hot output, that is the outputs are not correlated and can not be summed and thus derivative is reduced to the vector of single outputs
    # so np.sum (one element) / 1
    def d_val (self, y, l):
        # n = y.size 
        # return - np.sum (l-y) / n
        return y-l 

# -------------------------------------------------------- Test routines -----------------------------------------------------------------------------
# CE inspection on how it behaves if the right answer probability changes from 0,001% to 99,9%
# 10 classes, at first all prob = 0.001, then we change class 4 (wrong answer) from 99,9% to 0,001%
# and class 5 (right answer) from 0,001% to 99,9%
def PlotCE ():
    PosClassProb = 0.001
    label = np.full(shape=10, fill_value=0.01, dtype=np.float)
    label [4] = .99
    output = np.full(shape=10, fill_value=0.001, dtype=np.float)
    plt.title ('Cross Entropy response between 100% wrong and 100% right answer')
    plt.xlabel('x - right answer probability')
    plt.ylabel('y - cross entropy')

    # change the PosClassProb from 100% to 100% rigth answer
    while (PosClassProb < 0.99):
        out = output
        out [4] = PosClassProb
        out [5] = 1 - PosClassProb
        a = CrossEntropy(out, label)
        
        plt.scatter (PosClassProb, a, Color = 'red')    
        plt.pause (0.05)
        PosClassProb += 0.01





# PlotCE ()

# # 5th label is 1 and 6th output is 1
# label = np.full(shape=10, fill_value=0.01, dtype=np.float)
# label [4] = 0.99
# output = np.full(shape=10, fill_value=0.001, dtype=np.float)
# output [5] = 0.99

# a = CrossEntropy(output, label)
# print ('Label  : ', label)
# print ('Output : ', output)
# print ('MNIST ce output 100% wrong:', a)

# # 6th label is 1 and 6th output is 1
# label = np.full(shape=10, fill_value=0.01, dtype=np.float)
# label [5] = 0.99
# output = np.full(shape=10, fill_value=0.001, dtype=np.float)
# output [5] = 0.99

# a = CrossEntropy(output, label)
# print ('Label  : ', label)
# print ('Output : ', output)
# print ('MNIST ce output 100% right:', a)

# # 5th label is 1 and 5th output is 0.49 and 6th output is 0.5
# label = np.full(shape=10, fill_value=0.01, dtype=np.float)
# label [4] = 0.99
# output = np.full(shape=10, fill_value=0.001, dtype=np.float)
# output [4] = 0.49
# output [5] = 0.50

# a = CrossEntropy(output, label)
# print ('Label  : ', label)
# print ('Output : ', output)
# print ('MNIST ce output 50% wrong:', a)
