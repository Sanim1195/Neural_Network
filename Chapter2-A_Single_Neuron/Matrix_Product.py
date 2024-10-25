# 5 (Matrix Product and Transpose)
import numpy as np
# The matrix product  is an operation in which we have 2 matrices, and we are performing dot products of all combinations of rows from the first matrix and the columns of the 2nd matrix,resulting in a matrix of those atomic dot products

# IMP: To perform a matrix product, the size of the second dimension of the left (5,4) matrix must match the size of the first dimension of the right matrix (4,3).

# A row vector is a matrix whose first dimension’s size (the number of rows) equals 1 and the second dimension’s size (the number of columns) equals n​ — the vector size. In other words, it’s a 1×n array or array of shape (1, n):
# a = [a1,a2,a3....aN]


# with numpy and 3 values we would write it as:
vector = np.array([[1,2,3]])

# Note the use of double square brackets here. To transform a list into a matrix containing a single row (perform an equivalent operation of turning a vector into row vector), we can put it into a list and create numpy array:
a = [1,2,3]
print("Using numpy to transform a list into a matrix: ", np.array([a]), "\n")    #output: [[1 2 3]] which signifies a shape of (1,3). A 2D matrix with 1 row and 3 columns.


# Or we can turn it into a 1D array and expand dimensions using one of the NumPy abilities:
b = [1,2,3]
b_dims_expanded = np.expand_dims((b),axis=0)
print("Turining a list to a 1D array/matrix]: ", b_dims_expanded, "\n")   #output: [[1,2,3]]

# Imp: Transposition simply modifies a matrix in a way that its rows become columns and columns become rows.

# To transpose a row vector to a column vector using numpy
r = [1,2,3]     #row vector
c = [4,6,7]     # need to transpose this to a column vector to perfor dot product

c_transpose = np.array([c]).T
print("C Transposed(from row to column vector): \n", c_transpose , "\n")


#               A layer of neurons and Batch of  data  
# Now doing all that we learned so far using numoy, transpose and dot product :

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]    # shape (3,4)

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]] #shape(3,4)
 
biases = [2.0, 3.0, 0.5]

# Since the columns of inputs does not match with the rows of weights we have to perform a matrix transposition to match their  outer and inner shape dims.
layer_output = np.dot(inputs,np.array(weights).T) + biases
print("The layer output is: \n", layer_output, "\n")
