# %%
#  A layer of neurons
# He basically did the same thing but with a layer of neurons
# lets take 4 inputs and 3 neurons 
# every input  contains its own set of weights and its own bias, producing its own unique output


inputs = [1,2,3,2.5]

weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

outputs = [
    # Neuron 1
    inputs[0] * weights1[0] +
    inputs[1] * weights1[1] +
    inputs[2] * weights1[2] +
    inputs[3] * weights1[3] + bias1,
    
    # Neuron 2
    inputs[0] * weights2[0] +
    inputs[1] * weights2[1] +
    inputs[2] * weights2[2] +
    inputs[3] * weights2[3] + bias2,
    
    # Neuron 3
    inputs[0] * weights3[0] +
    inputs[1] * weights3[1] +
    inputs[2] * weights3[2] +
    inputs[3] * weights3[3] + bias3
]

print(outputs)
# %%
# The same thing as above can be done by having the weights as a list so we can iterate over them: 
# remember that every weiight is also represented as a list and has muktiple items in each weight
weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
bias = [2,3,0.5]
# output of current layer
layer_outputs = []
# for each neuron 
for neuron_weight,neuron_bias in zip(weights,bias):
    # Zeroed output of given neuron
    neuron_output = 0
    # for each input and weight to the neuron
    print(neuron_weight, ":" , neuron_bias)
    # for each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weight):
        # Multiply this input by assiciated weight
        # and add to the neurons output variable
        neuron_output = n_input * weight
        # now add bias to it
        neuron_output += neuron_bias
        # put neurons result to the layer's output list 
        layer_outputs.append(neuron_output)

print(layer_outputs)

# %%
