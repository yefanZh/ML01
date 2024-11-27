import numpy as np


def sigmoid_logistic(z):
    return 1/(1 + np.exp(-z))


points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])