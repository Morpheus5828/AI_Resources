import numpy as np

"""
In this file, we define classes representing activation functions to be applied between consecutive layers.
"""


class ReLU:

    name = "relu"

    @staticmethod
    def compute(x: np.array) -> np.array:
        return np.maximum(x, np.zeros(x.shape))

    @staticmethod
    def compute_derivative(x: np.array) -> np.array:
        return np.heaviside(x, np.zeros(x.shape))


class Sigmoid:

    name = "sigmoid"

    @staticmethod
    def compute(x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def compute_derivative(x: np.array) -> np.array:
        return Sigmoid.compute(x) * (1 - Sigmoid.compute(x))


class HyperbolicTangent:

    name = "tanh"

    @staticmethod
    def compute(x: np.array) -> np.array:
        return np.tanh(x)

    @staticmethod
    def compute_derivative(x: np.array) -> np.array:
        return 1 - np.square(np.tanh(x))

class Linear:

    name = "linear"

    @staticmethod
    def compute(x: np.array) -> np.array:
        return x

    @staticmethod
    def compute_derivative(x: np.array) -> np.array:
        return np.ones(shape=x.shape)


activation_functions = {
    Linear.name: Linear,
    ReLU.name: ReLU,
    Sigmoid.name: Sigmoid,
    HyperbolicTangent.name: HyperbolicTangent,
    #Softmax.name: Softmax
}
