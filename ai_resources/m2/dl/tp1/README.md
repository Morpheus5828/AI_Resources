# NNN_etu

This is a Numpy implementation of Neural Network for teaching purposes, which allows to design and train simple MLP           architectures.
 
Currently, implemented activations functions are linear, sigmoid, tanh, softmax. Costs functions are MSE and CE.              Optimization by minibatch SGD with weight decay.
 
The code is not complete, this is student version.
 
 ## Code structure

- cost_functions.py : mean squared error, cross entropy. Classes avec fonctions compute et derivative
- neural_network.py : fonctions fit, predict, evaluate, initialisation et d’autres intermédiaires dont forward et backprop qui appellent les forward/backward des différentes couches du réseau
- optimizers.py : pour le moment seulement SGD
- utils.py : save, load, one_hot encoding, ..
- layers/dense.py : fonctions forward, backward, initialisation, update 
- layers/activation_functions.py : linear, sigmoid, tanh, relu, softmax. Chacune définie par une classe avec deux fonctions : compute et derivative
- data/data_loader.py : télécharge mnist.pkl.gz, split, sous échantillonne le train, et met en forme
- main.py : définie l’architecture puis démarre l'entrainement

Pour lancer: `$ python main.py`


