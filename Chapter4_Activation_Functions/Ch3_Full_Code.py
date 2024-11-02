# This file contains all the code we have used in the chapter 2,3 and 4 so far

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Creating the first dense layer of neurons that takes an input from our dataset and performs dot product with our weights and adds the biases then outputs it.
# Consider matching the outer Dim shape of the dense layer(2,5) with that of the inner Dim of the input(10,2). That keeps our matrixes or tensors well suited for the dot product.

# A layer of neurons that takes it's input from either the inout or output from previous layers.   
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        # Initializing weights and bias
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1,n_neurons))
    
    def forward(self,inputs):
        # calculating output values from inputs, weights and biases and passing them forward
        self.output = np.dot(inputs,self.weights) + self.bias
    

# Mapping the non-linearity relationship our dataset 
class Activation_ReLU:
  """ ReLU for each dense layer in the network """  
  def forward(self, inputs) -> None:
        """ Activating the neurons and passing them forward: """
        self.output = np.maximum(0,inputs)

# For our output layer
class Softmax_Activation:
    """ Normalise the outputs gathered from the dense layers, Activating it using Softmax fnction(produces linearlity that can be used in classification tasks)and producing confidence score and the prediction   """
    def forward(self, inputs) -> None:
        # Getting unnormalised probabilities and controlling overflowing:
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        # normalise for each sample:
        probabilities = exp_values/np.sum(exp_values, axis=1,keepdims=True)
        self.output = probabilities

# Getting the dataset ready:
X,y = spiral_data(samples=100, classes=3)
print(np.shape(X))

#           Creating the architecture of the neural network

# creating a dense layer with 2 input neurons and 3 neurons for the 1st hidden layer:
dense1 = Layer_Dense(2,3)

# Creating ReLUactivation tobe used with the dense layer:
relu_activation1 =Activation_ReLU()

# Creating a second dense layer with 3 input features(as the output of previous dense layer here) and 3 output values(output values)
dense2 = Layer_Dense(3,3)

# Creating Softmax to be used with dense laye(output layer)
softmax_activation = Softmax_Activation()

#               Passing the dataset through the network
# Forward passing through the 1st dense layer
dense1.forward(X)
# Making the forward pass through the 1st Activation function(ReLu)
# It takes the output of first dense layer here as it's input
relu_activation1.forward(dense1.output)

# Making a forward pass through the second dense layer
#  it takes outputs of ReLu activation function of first layer as inputs
dense2.forward(relu_activation1.output)

# Making the forward pass through the Softmax activation function
# It takes the output of second dense layer here as it's inputs
softmax_activation.forward(dense2.output)

# checking the output results
print(softmax_activation.output[:5])


# pylint: disable=pointless-string-statement
""" We've completed what we need for forward-passing data through our model. We used the Rectified Linear (ReLU) activation function on the hidden layer, which works on a per-neuron basis.

We additionally used the Softmax activation function for the output layer since it accepts non-normalized values as input and outputs a probability distribution, which we're using as confidence scores for each class.

Recall that, although neurons are interconnected, they each have their respective weights and biases and are not “normalized” with each other. """ 

# To fix the randomness in our model we need use some loss function to check how wrong our model is and then tweak the biases and weights to decrease error over time