# based on Karpathy libs 
# https://github.com/karpathy/convnetjs/tree/master/demo/js/classify2d.js

import numpy as np
import matplotlib.pyplot as plt

# sigmoid
def sig (lgts):
    return 1 / (1 + np.exp (-lgts))

# hiperbolic tangens
def tsnh (lgts):
    # y = np.exp(2 * lgts)
    # return (y - 1) / (y + 1)
    return np.tanh (lgts)

# hiperbolic tangens derivative
def dtsnh (lgts):
    return 1 / np.square(np.cosh (lgts))

# softmax activation
def softmax (lgts):
    lgts = lgts - np.max (lgts)
    b = np.exp (lgts)
    b = b / np.sum (b, 0)
    np.clip (b, 0.01, 0.99, b) # clip the softmax result to comply with true output values in a loss function
    return b

# softmax derivative
def dsoftmax (lgts):
    s = softmax(lgts).reshape(-1,1)
    s1 = np.diagflat(s) - np.dot(s, s.T)
    return s1

def randomdata():
    x1 = np.random.rand (10, 1)
    y1 = np.square (x1) - np.random.randn (10,1)

    x2 = np.random.rand (10,1)
    y2 = - np.square (x2) + np.random.randn (10,1)

    plt.scatter (x1, y1, c='red')
    plt.scatter (x2, y2, c='green')

    plt.show()
    return data, label

# train 
def train(data, label):
    lr = 0.1
    iter = 70000

    acc = np.zeros (iter)
    for i in range (iter):
        for o in range (label.size):
            # forward
            ii = data [o].reshape (data [o].size, 1)
            
            hz = np.sum (np.dot (hw.T, ii), 1) + hb
            hz = hz.reshape (hz.size, 1)
            # ho = sig (hz) # sigmoid 
            # ho = hz * (hz > 0) # relu
            ho = tsnh (hz) # hiperbolic tangens
            np.clip (ho, 0.01, 0.99, ho) # does not work w/o clipping 
            
            oz = np.sum (np.dot (ow.T, ho), 1) + ob
            # oo = sig (oz).reshape (oz.size, 1) # sigmoid 
            oo = softmax (oz).reshape (oz.size, 1) 

            #backward
            dlo = oo - label [o] # loss function derivative
            # dao = sig (oz) * (1 - sig (oz)) # activation function derivative
            dao = dtsnh (oz)
            ddo = dlo * dao # delta
            dwo = np.dot (ddo, ho.T) # delta * logits (z) derivative = weight delta

            dlh = np.dot (ddo, ow.T) # "loss" of hidden layer: delta of output layer * weights of output layer
            # dah = sig (hz) * (1 - sig (hz)) # activation function derivative
            # dah = np.where (hz < 0, 0, 1)
            dah = dsoftmax (hz)  
            dwh = np.dot (dlh.T * dah, ii.T) # delta * logits (z) derivative = weight delta
            
            #update - gradient descent 
            ow = ow - lr * dwo.T
            hw = hw - lr * dwh.T
            acc [i] = acc [i] + np.abs (dlo)   


# draws learning squiggle 
def test(data, label):
    for o in range (label.size):
        # forward
        ii = input [o].reshape (input [o].size, 1)
        
        hz = np.sum (np.dot (hw.T, ii), 1) + hb
        hz = hz.reshape (hz.size, 1)
        # ho = sig (hz) # sigmoid 
        
        ho = hz * (hz > 0) # relu
        np.clip (ho, 0.01, 0.99, ho) 
        
        oz = np.sum (np.dot (ow.T, ho), 1) + ob
        oo = sig (oz) # sigmoid 
        
        print (oo - label [o])


# main
hw = np.random.randn (2, 6) # input 2 nuerons fc hidden 6 neurons  
hb = 0.23
ow = np.random.randn (6, 2) # softmax 
ob = 0.34

randomdata()
train (data, label)
test (data, label)

# test_x = np.array([[0], [2]])
# test_x_b = np.c_[np.ones((2,1)),test_x]
# test_y = np.dot (test_x_b, a)

# plt.plot(test_x, test_y, 'r-')
