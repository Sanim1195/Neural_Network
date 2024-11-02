# This file contains all the code we have used in the chapter 2,3 and 4 so far

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
X,y = spiral_data(samples=100, classes=3)
print(np.shape(X))

# Creating the first dense layer of neurons that takes an input from our dataset and performs dot product with our weights and adds the biases then outputs it.
# Consider matching the outer Dim shape of the dense layer(2,5) with that of the inner Dim of the input(10,2). That keeps our matrixes or tensors well suited for the dot product.

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        # Initializing weights and bias
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    