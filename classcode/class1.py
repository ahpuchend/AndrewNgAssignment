import numpy as np

a = np.array([1,2,3,4])
b = np.random.normal(size=[2,2])
a = np.dot(np.ones((1,2)) - [[1,2]],[[1],[2]])
print(a)