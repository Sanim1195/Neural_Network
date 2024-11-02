# 2
import numpy as np
# Rectified LInear Regression is mostly used to map non-linerar relationships(like waves) and functions

# Please consider reading the book page 15-31 for better understanding and visualizing the impact of tweaking weights and biases has on this activation function

#               For a list of values: 

# The ReLU in this code is a loop where we’re checking if the current value is greater than 0. If it is, we’re appending it to the output list, and if it’s not, we’re appending 0.

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

outputs = []

for i, value in enumerate(inputs):
    if inputs[i] > 0 :
        outputs.append(value)
    else:
        outputs.append(0)

print(outputs)


# The same code can be written with more clarity
# # We can basically just get the largest value between 0 and the neuron. For eg:

outputs2 = []

for value in inputs:
    outputs2.append(max(0,value))

print(outputs2)

# Numpy offers a similar method called maximum() that returns the largest value
outputs3 = np.maximum(0,inputs)
print("using numpy: ", outputs3)

class ReLU:

    def forward(self,input_s):
        """ Activating the neurons and passing them forward: """
        self.outputs = np.maximum(0,input_s)
