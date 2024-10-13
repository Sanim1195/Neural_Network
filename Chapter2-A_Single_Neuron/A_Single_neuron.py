#           For a single Neuron

#  when you initialize parameters in neural networks, our network will have weights initialized randomly, and biases set as zero to start
# The input will be either atual training data or the otputs of the previous layer

inputs = [1,2,3]
weights = [0.2, 0.8, -0.5]
# There is just one bias oer neuron
bias = 2 

# This neuron sums each input multiplied by that input’s weight, then adds the bias
output = (
    inputs[0]*weights[0] +
    inputs[1]*weights[1] +
    inputs[2]*weights[2] + bias
)

print(output)