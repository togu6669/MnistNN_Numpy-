# based on Karpathy libs
# https://github.com/karpathy/convnetjs/tree/master/demo/js/classify2d.js

# breast cancer dataset
# https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/
import numpy as np
import matplotlib.pyplot as plt


# hiperbolic tangens
def tsnh(lgts):
    # y = np.exp(2 * lgts)
    # return (y - 1) / (y + 1)
    return np.tanh(lgts)

# hiperbolic tangens derivative

def dtsnh(lgts):
    return 1 / np.square(np.cosh(lgts))

# softmax activation
def softmax(lgts):
    lgts = lgts - np.max(lgts)
    b = np.exp(lgts)
    out = b / np.sum(b, 0)
    # clip the softmax result to comply with true output values in a loss function
    np.clip(out, 0.01, 0.99, out)
    return out

# softmax derivative
def dsoftmax(lgts):
    s = softmax(lgts).reshape(-1, 1)
    out = np.diagflat(s) - np.dot(s, s.T)
    return out

# binary cross entropy / log loss and its derivative
# good study: https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
# another discussion: https://stats.stackexchange.com/questions/219241/gradient-for-logistic-loss-function#comment420534_219405
# next explanation: https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/
def binCE(y, l): 
    out = -(l*(np.log(y)) + (1-l)*(np.log(1-y))) # chain rule only for the 2nd element 
    return out

def dbinCE (y, l):
    out = (y-l) / y*(1-y)
    return out

def dbinCE2(y, l):
    z = np.zeros_like(y)
    # awful hack, we know that the value under the correct label index equals 1.00
    a = np.where(l == 1.00)
    if len(a) > 0 and len(a[0]) > 0:
        z[a[0]] = -1 / y[a[0]]
    return z

def randomdata():
    data = np.zeros((20, 2)) # [20, 2]
    data[:, 0] = np.random.rand(20, 1).reshape(20)

    data[:10, 1] = np.square(data[:10, 0]) - np.random.randn(10, 1).reshape(10)
    data[10:, 1] = - np.square(data[10:, 0]) + \
        np.random.randn(10, 1).reshape(10)

    label = np.zeros((2,20)) # [2, 20] one-hot encoded
    for i in range(10):
        label[0, i] = 1
        label[1, i+10] = 1
    return data, label

# train
def train(data, label, hw, hb, ow, ob):
    lr = 0.01
    iter = 7000

    acc = np.zeros(iter)
    for i in range(iter):
        for o in range(label.shape[1]-1):

            # forward
            ii = data[o].reshape(data[o].size, 1)

            hz = np.sum(np.dot(hw.T, ii), 1) + hb
            hz = hz.reshape(hz.size, 1)
            ho = tsnh(hz) # hiperbolic tangens

            oz = np.sum(np.dot(ow.T, ho), 1) + ob
            
            oo = softmax(oz) # oo = [2, 0]
            ol = binCE (oo, label[:,o]) 

            # backward
            # derivative of the loss fuction 
            dlo = dbinCE (oo, label[:,o]) # dlo = [2, ] if we have the label [1, 0] do we need the derivative of the 0 label? 
            dao = dsoftmax (oz) # derivative of the activation function (softmax), jacobian, oz = [2,] dao = [2, 2]
            ddo = np.dot (dlo.T, dao).reshape (dlo.size, 1) # output layer delta (used in the hidden layer), dot product ddo = [2, 1]
            # delta * logits (z) derivative (ho) = weight delta for the output layer (dwo)
            dwo = np.dot(ddo, ho.T) # ho = [12, 1] dwo = [2, 12]

            # "loss" of hidden layer: delta of output layer * weights of output layer
            dlh = np.dot(ddo.T, ow.T) # ow = [12, 2] dlh = [1, 12]
            dhz = dtsnh(hz) # derivative of the activation function (hiperbolic tangens), vector dhz = [12, 1]
            ddh = dlh.T * dhz # hidden layer delta, Hadamard, ddh = [12, 1]
            # delta * logits (z) derivative (ii) = weight delta for the hidden layer (dwh)
            dwh = np.dot(ddh, ii.T) # dwh = [12, 2]

            # update - stochastic gradient descent
            ow = ow - lr * dwo.T
            hw = hw - lr * dwh.T
            
            # accuracy: abs (oo[0] - label[o][0]) 
            err = np.round(np.abs (oo[0] - label[0, o]))
            if (err == 0.0): 
                acc[i] = acc[i] + 1
    return acc

# draws learning squiggle
def test(data, label, hw, hb, ow, ob):
    for o in range(label.shape[1]-1):
        # forward
        ii = data[o].reshape(data[o].size, 1)

        hz = np.sum(np.dot(hw.T, ii), 1) + hb
        hz = hz.reshape(hz.size, 1)
        ho = tsnh(hz)  # hiperbolic tangens

        oz = np.sum(np.dot(ow.T, ho), 1) + ob
        oo = softmax(oz)

        # maxv = 0.0
        # out = 0
        # # our output : the index of the most probable class, so 0 / 1
        # for o in range(oo.size):
        #     if (oo[o] > maxv):
        #         out = o
        #         maxv = oo[o]

        # print(maxv - label[o])


# main
hw1 = np.random.randn(2, 12)  # input 2 nuerons fc hidden 12 neurons
hb1 = 0.23
ow1 = np.random.randn(12, 2)  # softmax
ob1 = 0.34

dat, lab = randomdata()

# plt.scatter(dat[:10, 0], dat[:10, 1], c='red')
# plt.scatter(dat[10:, 0], dat[10:, 1], c='green')
# plt.show()

acc1 = train(dat, lab, hw1, hb1, ow1, ob1)
test(dat, lab, hw1, hb1, ow1, ob1)


# test_x = np.array([[0], [2]])
# test_x_b = np.c_[np.ones((2,1)),test_x]
# test_y = np.dot (test_x_b, a)

# plt.plot(test_x, test_y, 'r-')
