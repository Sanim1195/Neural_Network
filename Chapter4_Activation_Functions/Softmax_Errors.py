# 4
# Lets look at some prossible causes of exploding values (enormous numbers can wreak havoc down the line and render a network useless over time) which can happen by using softmax activation function

import numpy as np

# 1)           Exponential function overflowing:
# This happens when large numbers causes huge exponential values

print(np.exp(1000))     #Outouts: warning overflow encountered in exp

# As we know, the exponential function tends toward 0 as its input value approaches negative infinity, and the output is 1 when the input is 0

print(np.exp(-np.inf), np.exp(0))

# We can use the above property to prevent the exponential function from overflowing.
# np.inf refresents positive infinity whilst -np.inf represents negative infinity

# Suppose we subtract the maximum value from a list of input values. We would then change the output values to always be in a range from some negative value up to 0, as the largest number subtracted by itself returns 0, and any smaller number subtracted by it will result in a negative number — exactly the range discussed above.

# With Softmax, thanks to the normalization, we can subtract any value from all of the inputs, and it will not change the output:

# Use the above property inside the Softmax activation class
# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities (prevents overflowing)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
        self.output = probabilities

softmax = Activation_Softmax()
softmax.forward([[1,2,3]])
print(softmax.output)


# 2) Changed Confidences due to non-linearlity nature of exponentiation.
# To prevent this from hapenning we need to scale all of the input data to a neural network in the same way