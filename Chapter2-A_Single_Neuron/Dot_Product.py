# 4
import numpy as np

#  Traditionally, we use dot products for vectors
# When multiplying vectors, you either perform a dot product or a cross product.
# A cross product results in a vector while a dot product results in a scalar (a single value/number).


# A dot product of two vectors is a sum of products of consecutive vector elements. Both vectors must be of the same size (have an equal number of elements).

a = [1,2,3]
b = [2,3,4]
# Dot product
dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

print("Dot Product: ", dot_product, "\n")   #The output results as a scalar (i.e a single number)

# Vector addition: both vectors needs to be of the same size. The result will be a come a vector of this i.e a list

vector_addition = [a[0] + b[0], a[1] +b[1], a[2] + b[2]]
print("Vector addition: ", vector_addition,"\n") #The output is still a vector i.e a list 


# Now lets code a single neuron with Numpy
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

outputs = np.dot(inputs,weights) + bias
print("Single layer with numpy: ",outputs,"\n")


# Now lets try a layer of neurons with numpy 
inputs2 = [1.0, 2.0, 3.0, 2.5]
weights2 = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
biases2 = [2.0, 3.0, 0.5]

# Previously, we had calculated outputs of each neuron by performing a dot product and adding a bias, one by one. 
# Now we have changed the order of those operations — we’re performing dot product first as one operation on all neurons and inputs,
# and then we are adding a bias in the next operation. When we add two vectors using NumPy, each i-th element is added together, resulting in a new vector of the same size.

layer_outputs = np.dot(weights2,inputs2) + biases2
# here each list in weights2 is dot product of inputs2 then adds bias
# a dot product of a matrix and a vector results in a list of dot products

print("Dot Product of our matrix: ", layer_outputs)

