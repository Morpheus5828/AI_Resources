from cost_functions import cost_functions
from optimizers import optimizers
from layers.dense import Dense
from utils import generate_batches

import numpy as np
from pathlib import Path
from time import perf_counter


class NeuralNetwork:

    def __init__(self,
                 architecture=None,
                 optimizer="sgd",
                 cost_function="mse"
                 ) -> None:

        # ----- CLASSIFICATION ATTRIBUTES -----
        self.layers: list = []
        self.classes: dict = None

        # ----- HYPER-PARAMETERS -----
        self.batch_size: int = 32
        self.number_of_epochs: int = 10
        self.optimizer = optimizers[optimizer]()
        self.cost_function = cost_functions[cost_function]()

        self.initialize_architecture(architecture)

    
    def fit(self,
              x: np.array,
              y: np.array,
              validation_data: np.array = None,
              validation_data_labels: np.array = None,
              learning_rate: float = None,
              epochs: int = 5,
              batch_size: int = 50) -> None:
        """
        This method trains the network. It takes an input vector and a label vector.
        If no validation data is passed, the method keeps 10% of the training data to serve as validation data.

        :param x: an array of input vectors
        :param y: an array of label vectors
        :param validation_data: an array of input vectors to measure the network validation accuracy throughout the
        training session
        :param validation_data_labels: an array of label vectors to measure the network validation accuracy throughout
        the training session
        :param learning_rate: a float used to define the learning rate of the optimizer
        :param epochs: redefine the number of epochs for the network to be trained on
        :param batch size: redefine the size of the batches use for training
        :return: None
        """

        # set class attributes
        self.number_of_epochs = epochs
        self.batch_size = batch_size


        # sets the learning of the optimizer if passed as an argument
        if learning_rate is not None:
            self.optimizer.learning_rate = learning_rate
        
        # create batches iterator as a generator
        batches = generate_batches(x, y, self.batch_size)
        
        ############                ############
        ############ begin training ############
        ############                ############
        
        # looping over epochs
        for epoch in range(self.number_of_epochs):

            print(f"\n--- Epoch: {epoch + 1} ---\n")
            print("Progress:")

            timings = []

            # looping over batches
            for batch in range(x.shape[0] // self.batch_size):

                start = perf_counter()

                # get a random batch
                x_batch, y_batch = next(batches)

                # calculate output of the network for the given input
                prediction = self.forward(x=x_batch)

                # backpropagate the error through the whole network
                self.backpropagation(y=y_batch, expected_output=prediction)
                
                # update the weights and biases of the network
                self.update_parameters()
                
                timings.append(perf_counter() - start)

                if batch % 100 == 0:
                    # Monitoring training session
                    print(f"Batch {batch} / {x.shape[0] // self.batch_size}, "
                          f"accuracy: {100 * self.evaluate(x, y):.2f}%, "
                          f"validation accuracy: {100 * self.evaluate(validation_data, validation_data_labels):.2f}%, "
                          f"mean execution time per batch: {1000 * np.mean(timings):.2f} ms",
                          end="\r")

            print("\n", end="")
        
    def forward(self, x: np.array) -> np.array:
        """
        This method computes the output of the network given an array of vectors.

        :param x: an array of size batch_size input vectors
        :return: the array of size batch_size containing in position i the output of the input vector x[i]
        """

        x_inferred = x

        for layer in self.layers:
            x_inferred = layer.propagate_forward(x_inferred)

        return x_inferred

    def backpropagation(self, y: np.array, expected_output: np.array) -> None:
        """
        This method computes the error that propagates through each layer of the network.

        :param y: an array of size batch_size containing output vectors computed through inference
        :param expected_output: an array of size batch_size containing the desired output for each input vector
        :return: None
        """
        
        # compute deltaE (TD notation)
        deltaE =  (expected_output - y) / y.shape[0]
        # compute delta_L, the error to propagate from last layer
        deltaL = np.multiply(deltaE, expected_output)
        # call recursive deltas from before last to first
        for layer in self.layers[-2::-1]:
            deltaL = layer.propagate_backward(deltaL)

    def update_parameters(self) -> None:
        """
        This method performs the gradient descent algorithm by calling the weights and biases update method of each
        layer.

        :return: None
        """

        for layer in self.layers:
            layer.update_weights(self.optimizer)
            layer.update_biases(self.optimizer)

    def evaluate(self, x: np.array, y: np.array) -> float:
        """
        This method gives a sense of measure of the classification accuracy of the neural network.

        :param x: an array of input vectors
        :param y: an array of label vectors
        :return: a float between 0 an 1 measuring the accuracy of the network
        """
        return np.linalg.norm(self.forward(x) - y) ** 2

    def predict(self, x: np.array) -> np.array:
        """
        Computes the output for each input vector of the x parameter and returns what class each of the input vectors
        was assigned to only if the "classes" key was included in the architecture provided.

        :param x: an array of input vectors
        :return: an array which i-th element is the class x[i] was assigned to
        """
        prediction= self.forward(x)
        return np.argmax(prediction)


    def initialize_architecture(self, architecture: dict) -> None:
        """
        This method is responsible for the creation of the neural network per se.

        :param architecture: a dictionary to define the architecture of the neural network. Acceptable keys are :
            - model: a list of dictionary defining sequentially each layer of the network
            - classes: a list that allows to map the index of a label vector to the actual class it encodes (e.g, ["dog", "cat"])
        :return:
        """
        
        layers = architecture["layers"]
        
        for i in range(1, len(layers)):
            layer = layers[i]
            
            if layer["type"] == "dense":
                layer["input neurons"] = layers[i-1]["output neurons"]
                self.layers.append(Dense(**layer))
                self.layers[-1].init_layer() # init weights
        
        self.classes = architecture["classes"]

