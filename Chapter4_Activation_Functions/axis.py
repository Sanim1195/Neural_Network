# Understanding axis
import numpy as np

#Let us consider the following matrix as a batch data: 
matrix2D =[
    [1,2,3,5],
    [3,5,7,6],
    [1,12,1,2]
]
print("The shape of matrix2D is: ",np.shape(matrix2D), "\n")

# For a 2D matrix(3,4) Axis 0 (which is the 3 rows) runs vertically down the rows. So, when you sum along axis 0, you’re summing each element in the same column across all rows.
# In other words, the output has the same size as this axis, as at each of the positions of this output, the values from all the other dimensions at this position are summed to form it.
# “For simplicity, consider that axis=0 sums the columns, resulting in a row vector, and axis=1 sums the rows, resulting in a column vector if keepdims=True.”

axis0 = np.sum(matrix2D ,axis=0, keepdims=True)
print("Axis 0: ", axis0, "\n","The shape is: ", np.shape(axis0))

axis1 = np.sum(matrix2D ,axis=1, keepdims=True)
print("Axis 1: ",axis1, "The shape is: ", np.shape(axis1))



 