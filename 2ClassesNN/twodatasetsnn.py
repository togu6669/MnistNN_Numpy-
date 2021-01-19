# based on Karpathy libs
# https://github.com/karpathy/convnetjs/tree/master/demo/js/classify2d.js

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
    b = b / np.sum(b, 0)
    # clip the softmax result to comply with true output values in a loss function
    np.clip(b, 0.01, 0.99, b)
    return b

# softmax derivative


def dsoftmax(lgts):
    s = softmax(lgts).reshape(-1, 1)
    s1 = np.diagflat(s) - np.dot(s, s.T)
    return s1


def randomdata():
    data = np.zeros((20, 2))
    data[:, 0] = np.random.rand(20, 1).reshape(20)

    data[:10, 1] = np.square(data[:10, 0]) - np.random.randn(10, 1).reshape(10)
    data[10:, 1] = - np.square(data[10:, 0]) + \
        np.random.randn(10, 1).reshape(10)

    label = np.zeros(20)
    for i in range(10):
        label[i+10] = 1
    return data, label

# train
def train(data, label, hw, hb, ow, ob):
    lr = 0.1
    iter = 70000

    acc = np.zeros(iter)
    for i in range(iter):
        for o in range(label.size):
            # forward
            ii = data[o].reshape(data[o].size, 1)

            hz = np.sum(np.dot(hw.T, ii), 1) + hb
            hz = hz.reshape(hz.size, 1)
            ho = tsnh(hz)  # hiperbolic tangens

            oz = np.sum(np.dot(ow.T, ho), 1) + ob
            oo = softmax(oz).reshape(oz.size, 1)

                       
            # max function derivative: maxv for the max class, 0 for the other
            maxv = 0.0
            out = 0
            # we have two classes so our derivative output : the index of the most probable class, so 0 / 1
            for o in range(oo.size):
                if (oo[o] > maxv):
                    out = o
                    maxv = oo[o]

            # backward
            dlo = maxv - label[o]  # loss function derivative, if  maxv about 1  and label [class 1]= 1 then dlo is about 0  
            dao = dsoftmax(oz)
            ddo = dlo * dao  # delta
            # delta * logits (z) derivative = weight delta
            dwo = np.dot(ddo, ho.T)

            # "loss" of hidden layer: delta of output layer * weights of output layer
            dlh = np.dot(ddo, ow.T)
            dah = dtsnh(hz)
            # delta * logits (z) derivative = weight delta
            dwh = np.dot(dlh.T * dah, ii.T)

            # update - gradient descent
            ow = ow - lr * dwo.T
            hw = hw - lr * dwh.T
            acc[i] = acc[i] + np.abs(dlo)


# draws learning squiggle
def test(data, label, hw, hb, ow, ob):
    for o in range(label.size):
        # forward
        ii = data[o].reshape(data[o].size, 1)

        hz = np.sum(np.dot(hw.T, ii), 1) + hb
        hz = hz.reshape(hz.size, 1)
        ho = tsnh(hz)  # hiperbolic tangens

        oz = np.sum(np.dot(ow.T, ho), 1) + ob
        oo = softmax(oz).reshape(oz.size, 1)

        maxv = 0.0
        out = 0
        # our output : the index of the most probable class, so 0 / 1
        for o in range(oo.size):
            if (oo[o] > maxv):
                out = o
                maxv = oo[o]

        print(maxv - label[o])


# main
hw1 = np.random.randn(2, 12)  # input 2 nuerons fc hidden 12 neurons
hb1 = 0.23
ow1 = np.random.randn(12, 2)  # softmax
ob1 = 0.34

dat, lab = randomdata()

plt.scatter(dat[:10, 0], dat[:10, 1], c='red')
plt.scatter(dat[10:, 0], dat[10:, 1], c='green')
plt.show()

train(dat, lab, hw1, hb1, ow1, ob1)
test(dat, lab, hw1, hb1, ow1, ob1)


# test_x = np.array([[0], [2]])
# test_x_b = np.c_[np.ones((2,1)),test_x]
# test_y = np.dot (test_x_b, a)

# plt.plot(test_x, test_y, 'r-')
