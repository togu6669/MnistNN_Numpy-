import numpy as np
import matplotlib.pyplot as plt

def sig (lgts):
    return 1 / (1 + np.exp (-lgts))

x1 = np.random.rand (10,1)
y1 = np.square (x1) - np.random.randn (10,1)

x2 = np.random.rand (10,1)
y2 = - np.square (x2) + np.random.randn (10,1)

plt.scatter (x1, y1, c='red')
plt.scatter (x2, y2, c='green')

# test_x = np.array([[0], [2]])
# test_x_b = np.c_[np.ones((2,1)),test_x]
# test_y = np.dot (test_x_b, a)

# plt.plot(test_x, test_y, 'r-')
plt.show()
