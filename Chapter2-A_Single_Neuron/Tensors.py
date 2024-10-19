import numpy as np


# Tensors are closely related to arrays

# This is a list
list1 = [1,2,3,4,5]
print(f"shape of list1:{np.shape(list1)} ")

# This is a list of lists 👇
lol = [[1,2,3],
       [4,3,2]]
print(f"shape of lol :{np.shape(lol)} ")
# Th output(2,3) means its a 2D matrix with 2 rows and 3 columns

# This is a list of lists of lists 👇
lolol = [[[1,2,3,4],
          [2,3,6,7]],
         [[4,5,6,7],
          [4,8,5,4]],
         [[0,8,6,5],
          [6,7,2,9]]]
print(f"shape of lolol :{np.shape(lolol)} ")
# The output for the above shape (3,2,4) means it's a 3D matrix with 3 layers of 2D matrix with 2 rows and 4 columns 

# Everything shown so far could also be an array or an array representation of a tensor.

another_list_of_lists = [[4,2,3],[5,1]]
# The above list of lists cannot be an array because it is not homologous
# A list of lists is homologous if each list along a dimension is identically long,
# and this must be true for each dimension



# What is a Tensor??
# A tensor object is an object that can be represented as an array

