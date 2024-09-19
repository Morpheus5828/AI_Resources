from layers.activation_functions import activation_functions

import math
import numpy as np
from prompt_toolkit.key_binding.bindings.named_commands import self_insert


class Dense:

    number_of_dense_layers = 0

    def __init__(self, **kwargs):

        Dense.number_of_dense_layers += 1
        #self.name = f"Dense_{Dense.number_of_dense_layers}"
        
        self.name = kwargs.get("name", "dense"+str(Dense.number_of_dense_layers))
        self.input_neurons: int = kwargs.get("input neurons", 0) # input dimension
        self.output_neurons: int = kwargs.get("output neurons", 1) # output dimension

        self.x: np.array = None # input vector
        self.a: np.array = None # pre activation neurons
        self.z: np.array = None # post activation neurons

        self.weights: np.array = None # weights parameters, matrix
        self.biases: np.array = None # biases parameters, vector

        self.weights_gradient: np.array = None # weights gradients (loss wrt weights), matrix
        self.biases_gradient: np.array = None # biases gradients (loss wrt biases), vector

        # add activation function within the layer
        self.activation_function = activation_functions[kwargs.get("activation function", "linear")]


    def propagate_forward(self, x: np.array) -> np.array:
        """
        This method computes the output of the layer for a given input.

        :param x: an array of input vectors
        :return: an array containing the output vector for each input vector
        """

        self.x = x
        self.a = x @ self.weights.T + self.biases
        self.z = self.activation_function.compute(self.a)
        return self.z

    def propagate_backward(self, error_from_next: np.array) -> np.array:
        """
        Given the error from the next layer (one step closer to the network output), compute delta_i and updates the weights and biases gradient then returns the error to be propagated in the precedent layer (one step closer to the network input). Check the recursive part of deltas computation to see what this function return.

        :param error_from_next: an array containing the error computed at the next layer
        :return: an array containing the error to be propagated deeper in the network
        """
        delta_i = error_from_next * self.activation_function.compute_derivative(self.a)

        # Compute gradients for weights and biases
        dW = np.einsum('bi,bj->bij', delta_i, self.x)
        self.weights_gradient = np.mean(dW, axis=0)
        self.biases_gradient = np.mean(delta_i, axis=0)

        return delta_i @ self.weights

    def update_weights(self, optimizer) -> None:
        """
        Given the neural network optimizer, this method optimizes the weights parameters of this layer.

        :param optimizer: Optimizer subclass, strategy used for the neural network parameters to be updated
        :return: None
        """

        self.weights = optimizer.optimize(self.weights, self.weights_gradient)
        self.weights_gradient = np.zeros(self.weights.shape)

    def update_biases(self, optimizer) -> None:
        """
        Given the neural network optimizer, this method optimizes the biases parameters of this layer.

        :param optimizer: Optimizer subclass, strategy used for the neural network parameters to be updated
        :return: None
        """

        self.biases = optimizer.optimize(self.biases, self.biases_gradient)
        self.biases_gradient = np.zeros(self.biases.shape)

    def init_layer(self) -> None:
        """
        Initializes the parameters of this layer.

        :param input_neurons: size of the input vectors that will be passed to the layer
        :return: None
        """

        self.x = np.zeros((self.input_neurons, ))
        self.a = np.zeros((self.output_neurons, ))
        self.z = np.zeros((self.output_neurons, ))

        self.init_weights()
        self.biases = np.zeros((self.output_neurons, ))

        self.weights_gradient = np.zeros(self.weights.shape)
        self.biases_gradient = np.zeros(self.biases.shape)

    def init_weights(self) -> None:
        """
        Depending on the activation function assigned to the layer, we use different strategies to initialize the weights matrix.

        See: https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init for more information.

        :return: None
        """
        self.weights = np.random.sample((self.output_neurons, self.input_neurons)) * 2 - 1

        r = math.sqrt(6 / (self.input_neurons + self.output_neurons))

        if self.activation_function.name == "relu":
            self.weights *= math.sqrt(2) * math.sqrt(6 / (self.input_neurons + self.output_neurons))

        elif self.activation_function.name == "sigmoid":
            self.weights *= 4 * r

        elif self.activation_function.name == "tanh":
            self.weights *= r
        
        elif self.activation_function.name == "linear":
            self.weights *= r
            
        elif self.activation_function.name == "softmax":
            self.weights *= 4 * r
