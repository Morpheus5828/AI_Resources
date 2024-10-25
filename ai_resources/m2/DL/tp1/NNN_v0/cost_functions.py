import numpy as np

"""
Define classes for cost functions 
"""


class MeanSquaredError:

    name = "mse"
    
    @staticmethod
    def compute(x, y) -> float:

        n = x.shape[0]
        return np.sum(np.sum(np.square(x - y), axis=1)) / (2 * n)

    @staticmethod
    def compute_derivative(x: np.array, y: np.array) -> np.array:

        return x - y

cost_functions = {
    MeanSquaredError.name: MeanSquaredError,
    #CrossEntropy.name: CrossEntropy
}
