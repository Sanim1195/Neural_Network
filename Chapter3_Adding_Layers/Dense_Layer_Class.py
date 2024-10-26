# 2
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


# So far we have only used a Dense Layer or a fully connected layer.
# Our dense layer will begin with a Dense class

# np.random.randn and np.zeros: These methods are convenient ways to initialize arrays. np.random.randn produces a Gaussian distribution with a mean of 0 and a variance of 1, which means that it’ll generate random numbers, positive and negative, centered at 0 and with the mean value close to 0. In general, neural networks work best with values between -1 and +1,
nnfs.init()

# Dense Layer
class Layer_Dense:
    # Layer Initilaization
    def __init__(self, n_inputs, n_neurons) -> None:
        # The weights here will be the number of inputs for the first dimension and the number of neurons for the 2nd dimension.
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        # We’ll initialize the biases with the shape of (1, n_neurons), as a row vector, which will let us easily add it to the result of the dot product later, without additional operations like transposition.
        self.biases = np.zeros((1,n_neurons))
        print("Weights: \n", self.weights, "\n")
        print("Biases: \n", self.biases, "\n")
    
    # Forward Pass
    def forward(self,inputs) -> None:
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs,self.weights) + self.biases

# initializing dataset
X,y = spiral_data(samples=100, classes=3)

# creating a dense layer with 2 inputs and 3 output values(neurons)
dense1 = Layer_Dense(2,3)

# performing forward pass of our training data through this layer
dense1.forward(X)
print(dense1.output[:5])




