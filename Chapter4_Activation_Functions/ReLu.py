# 2
# Just playing with a basic implemenatation here. Check ReLU_Activation.py for descriptive implemantation.

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

inputs2 = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
outputs2 = []

for i, value in enumerate(inputs2):
    outputs2.append(max(0,value))

print(outputs2)

# Numpy offers a similar method called maximum() that returns the largest value
