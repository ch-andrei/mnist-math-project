import numpy as np
from NeuralNetwork import NeuralNetwork


# Rounds the input to either a 1 or a 0
# x: number in the range [0, 1]
# Output: 1 for x in [0.5, 1] and 0 for x in [0, 0.5)
def binarizer(x):
    def f(value): return int(round(value,0))
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return [f(i) for i in x]
    else:
        return f(x)


print '8-bit identity'
X = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]])
y = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]])
nn = NeuralNetwork(8, [5,5], 8)
nn.train(X, y, 100000, 0.3)
print nn.predictOutput(X, binarizer)

print '3-bit AND'
X = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
y = np.array([[0],[0],[0],[0],[0],[0],[0],[1]])
nn = NeuralNetwork(3, [10,10], 1)
nn.train(X, y, 100000, 0.3)
print nn.predictOutput(X, binarizer)