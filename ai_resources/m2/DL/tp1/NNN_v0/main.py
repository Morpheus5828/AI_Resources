from utils import load_network, save_network
from neural_network import NeuralNetwork
from data import data_loader

import numpy as np
import matplotlib.pyplot as plt

# Data loading and formatting
x_train, y_train, x_val, y_val, x_test, y_test = data_loader.load_data()


# Define neural network architecture as a dictionary with keys:
# - "layers", a list of layers, each as a dictionary
# - "classes", used to define the classes as a list (e.g: ["car", "bike"]).
architecture = {
    "layers": [
        {
            "name": "input",
            "type": "input",
            "output neurons": x_train[0].shape[0],
        },
        {
            "name": "hidden1",
            "type": "dense",
            "output neurons": 100,
            "activation function": "relu" #"sigmoid"
        },
        {
            "name": "output",
            "type": "dense",
            "output neurons": 10,
            "activation function": "linear" #"linear" "softmax"
        }
    ],
    "classes": [i for i in range(10)]  
}

# train model using SGD and MSE
neural_network = NeuralNetwork(architecture=architecture, optimizer="sgd", cost_function="mse")

neural_network.fit(x_train,
                     y_train,
                     validation_data=x_val[:1000],
                     validation_data_labels=y_val[:1000],
                     learning_rate=0.01,
                     epochs=10,
                     batch_size=12)

save_network(neural_network, path="my_model")

# Loads a neural network
neural_network = load_network("my_model")

# Evaluate and show a single test example
print("\nAccuracy on test dataset:", neural_network.evaluate(x_test, y_test) * 100.)

plt.imshow(np.reshape(x_test[100], (28, 28)), cmap="gray")
plt.title("predicted as " + str(neural_network.predict(x_test[100])))
plt.show()
# Saves the neural network

