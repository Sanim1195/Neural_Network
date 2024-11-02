# 3

# The Softmax function is used to convert a vector of raw scores (logits) into probabilities. It ensures that the output probabilities sum to 1, making it useful for classification tasks. It is mostlty used in the network's output layer to return a normalized distribution of probabilities for our raw scores or outputs (These raw scores or outputs are non-linear in shape) received from the neuron's hidden layers.

# The returned distribution from the softmax activation represents the Confidence Scores for each class and will add up to 1. eg: If our neural network had to predict a class of A or B based on a user input:  The output after applying Softmax Activation would be something like :  [0.45,0.55], which reperesents a confidence level of (0.45) for class A and a confidence level of (0.55) for class B. The sum of these 2 confidence level sums up to 1.

# The predictibility of the model depends upon the highest confidence score. Which in our above example would be class B with the confidence score of 0.55. This function is great for Probability Distribution and Multi-Class Classification tasks.

# Formula:
# The softmax function for a given number zi,j [where zi,j is the Raw score for the ith indexed j class] =  the Exponential of zi,j (the current raw score) DIVIDED BY the ∑(summation) of the exponentiated raw scores for all classes in given sample.


# Output froma single neuron from previous layers of neuron with 3 classes
outputs = [4.8, 1.21, 2.385]

# Mathematical consonant e
E = 2.71828182846
exponentiated = []

for output in outputs:
    exponentiated.append(E ** output)

print(f"Exponentiated values: {exponentiated} ", "\n")

#               Exponential serves multiple purposes:
# 1)  To calculate the probabilities, we need non-negative values.
# 2) A negative probability (or confidence) does not make much sense. An exponential value of any number is always non-negative — it returns 0 for negative infinity, 1 for the input of 0, and increases for positive values.

# The exponential function is a monotonic function (with higher input values, outputs are also higher)

# Once exponentiated we want to convert these numbers to a probability distribution (converting the values into the vector of confidences, one for each class, which add up to 1 for everything in the (vector).
# What that means is that we’re about to perform a normalization where we take a given value and divide it by the sum of all of the values.
# For our outputs, exponentiated at this stage,that’s what the equation of the Softmax function describes next — to take a given exponentiated value and divide it by the sum of all of the exponentiated values.

# Normalizing the code:
norm_base = sum(exponentiated) #summing the exponentiated values
print("The sum of all exponentiated base: " , norm_base, "\n")
normalized_value = []

for value in exponentiated:
    normalized_value.append(value/norm_base)

print(f"Normalized Exponentiatd values: {normalized_value}", "\n")



# we can do the same operation using numpy and it's methods
import numpy as np

layer_outputs = np.array([
    [4.8, 1.21, 2.385],
    [8.9, -1.81, 0.2],
    [1.41, 1.051, 0.026]
])
print("The shape of layer outputs: ", np.shape(layer_outputs), "\n")

# For each  vetor calculate the exponential value
exp_values = np.exp(layer_outputs)
print("Exponentiated values using np.exp(): \n" ,exp_values , "\n")

# In a 2D array/matrix, axis 0 refers to the rows, and axis 1 refers to the columns
# So in order to get the sum of each row in a 2d matrix and keep dims to keep the same dimension i.e a 2D matrix. 

sum_of_exp_values = np.sum(layer_outputs, axis=0, keepdims=True)
print("The sum of exponentiated values along axis=0 (sum along the rows) are : \n",sum_of_exp_values, "\n")

# Without specifying the axis the sum function will sum up all values inside the array
print("Using sum without specifying the axis results in : ", np.sum(sum_of_exp_values), "\n")

# But this is not what we want: We want to sum all the columns(class). To which we juat use the axis= 1. Remember to keep the optput's dimension equal to the input dimension
sum_of_exp_values2 = np.sum(layer_outputs, axis=1, keepdims=True)
print("The sum of exponentiated values along axis=1 (sum along the columns) are : \n",sum_of_exp_values2, "\n")

# Normalizing the values(which means dividng the exp value with the sum of all exp classes).
norm_value = exp_values/sum_of_exp_values2
print(norm_value)

