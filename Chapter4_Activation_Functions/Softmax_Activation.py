# 3

# The Softmax function is used to convert a vector of raw scores (logits) into probabilities. It ensures that the output probabilities sum to 1, making it useful for classification tasks. It is mostlty used in the network's output layer to return a normalized distribution of probabilities for our raw scores or outputs (These raw scores or outputs are non-linear in shape) received from the neuron's hidden layers.

# The returned distribution from the softmax activation represents the Confidence Scores for each class and will add up to 1. eg: If our neural network had to predict a class of A or B based on a user input:  The output after applying Softmax Activation would be something like :  [0.45,0.55], which reperesents a confidence level of (0.45) for class A and a confidence level of (0.55) for class B. The sum of these 2 confidence level sums up to 1.

# The predictibility of the model depends upon the highest confidence score. Which in our above example would be class B with the confidence score of 0.55. This function is great for Probability Distribution and Multi-Class Classification tasks.

# Formula:
# The softmax function for a given number zi,j [where zi,j is the Raw score for the ith indexed j class] =  the Exponential of zi,j (the current raw score) DIVIDED BY the Summation of the exponentiated raw scores for all classes in given sample.


# Output from previous layer of neuron with 3 classes
outputs = [4.8, 1.21, 2.385]

# Mathematical consonant e
E = 2.71828182846
confidence = []
value = 0

for raw_scores in outputs:
    for j, output in enumerate(outputs):
        value += (E ** output)
    confidence.append((E ** raw_scores) / value)

print(confidence)
