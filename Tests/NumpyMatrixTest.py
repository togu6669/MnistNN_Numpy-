import numpy as np


a = np.array ([[1, 2, 3, 5], [10, 20, 30, 50], [100, 200, 300, 500]])
b = np.array ([5, 10, 20])

c = b.dot (a)

print (c)