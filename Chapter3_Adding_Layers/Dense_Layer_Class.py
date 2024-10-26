# 2
import numpy as np

# So far we have only used a Dense Layer or a fully connected layer.
# Our dense layer will begin with a Dense class

# np.random.randn and np.zeros: These methods are convenient ways to initialize arrays. np.random.randn ​produces a Gaussian distribution with a mean of 0 and a variance of 1, which means that it’ll generate random numbers, positive and negative, centered at 0 and with the mean value close to 0. In general, neural networks work best with values between -1 and +1,

class Layer_Dense:
    # Layer Initilaization
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.01 * np.ramdom.randn(n_inputs,n_neurons)
        self.bias = np.zeros((1,n_neurons))