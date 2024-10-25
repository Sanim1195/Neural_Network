# 1
import numpy as np

#  To havehidden layers, we must make sure that the expected input to that layer matches the previous layer’s output. We have set the number of neurons in a layer by setting how many weight sets and biases we have. The previous layer’s influence on weight sets for the current layer is that each weight set needs to have a separate weight per input. This means a distinct weight per neuron from the previous layer (or feature if we’re talking the input).

inputs = [[1, 2, 3, 2.5],
          [2., 5., -1., 2],
          [-1.5, 2.7, 3.3, -0.8]]
print("The shape of inputs is: ", np.shape(inputs), "\n")  #outputs: (3,4)

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
print("The shape of weights is: ", np.shape(weights), "\n") #outputs: (3,4)
# Since the shapes are not aligned for dot product, we need to transpose the weights matrix

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]   #shape: (3,3)

biases2 = [-1, 2, -0.5]

# calculating for input layer:
layer1_outputs= np.dot(inputs,np.array(weights).T) + biases
print("Layer1_outputs: \n", layer1_outputs, "\n")
print("Shape of layer1_outputs: ", np.shape(layer1_outputs), "\n")

# calculating for our hidden layer where we use the output of our input layer as the input to our hidden layer's neurons
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print("Layer2_outputs: \n", layer2_outputs, "\n")
