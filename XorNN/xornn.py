import numpy as np
import matplotlib.pyplot as plt

def sig (lgts):
    return 1 / (1 + np.exp (-lgts))

input = np.array ([[0, 0],
                   [1, 0], 
                   [0, 1], 
                   [1, 1]])

label = np.array ([0, 1, 1, 0])

hw = np.random.randn (2, 3)
hb = 0.23
ow = np.random.randn (3, 1)
ob = 0.34

# train 
lr = 0.05
iter = 60000
for i in range (iter):
    for o in range (label.size):
        # forward
        ii = input [o].reshape (input [o].size, 1)
        
        hz = np.sum (np.dot (hw.T, ii), 1) + hb
        hz = hz.reshape (hz.size, 1)
        ho = sig (hz) # sigmoid 
        # ho = hz * (hz > 0) # relu
        # np.clip (ho, 0.01, 0.99, ho)  # ? 
        oz = np.sum (np.dot (ow.T, ho), 1) + ob
        oo = sig (oz).reshape (oz.size, 1) # sigmoid 

        #backward
        dlo = oo - label [o] # loss function derivative
        dao = sig (oz) * (1 - sig (oz)) # activation function derivative
        ddo = dlo * dao # delta
        dwo = np.dot (ddo, ho.T) # delta * logits (z) derivative = weight delta

        dlh = np.dot (ddo, ow.T) # "loss" of hidden layer: delta of output layer * weights of output layer
        dah = sig (hz) * (1 - sig (hz)) # activation function derivative
        # dah = np.where (hz < 0, 0, 1)  
        dwh = np.dot (dlh.T * dah, ii.T) # delta * logits (z) derivative = weight delta
        
        #update - gradient descent 
        ow = ow - lr * dwo.T
        hw = hw - lr * dwh.T 

# test
for o in range (label.size):
    # forward
    ii = input [o].reshape (input [o].size, 1)
    
    hz = np.sum (np.dot (hw.T, ii), 1) + hb
    hz = hz.reshape (hz.size, 1)
    ho = sig (hz) # sigmoid 
    
    # ho = hz * (hz > 0) # relu
    # np.clip (ho, 0.01, 0.99, ho)  # ? 
    
    oz = np.sum (np.dot (ow.T, ho), 1) + ob
    oo = sig (oz) # sigmoid 
    
    print (oo - label [o])