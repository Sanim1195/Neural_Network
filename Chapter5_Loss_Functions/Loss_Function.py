# 1.
import math

#            Categorical Cross Entropy Loss Function:
# Categorical cross-entropy is explicitly used to compare a “ground-truth” probability (y or "targets") and some predicted distribution (y-hat or "predictions")


# pylint: disable=pointless-string-statement,line-too-long
""" 
    The formula for calculating the categorical cross-entropy of y(actual/desired distribution) and y-hat (predicted distribution) is: 
                Li =  Negative Summation of Yijlog(y-Hatij)

    Where Li denotes sample loss value,
    i is the i-th sample in the set,
    j is the label/output index, 
    y denotes the target values, and y-hat denotes the predicted values. 
"""

# The output we receive from a softmax activation is a vector like eg: [0.7, 0.1, 0.2]. We can also represent the current prediction as [1, 0, 0]. Keeping one for the desired calss and the rest to 0. This is also known as one hot encoding
# During the loss error claculation, the 0's in the one-hot encoded vector, when multiplied with it's corresponding value results to a 0. Using one hot encoding we can make the calculation of the error function relatively simpler.

softmax_outputs = [0.7, 0.1, 0.2]
# Ground truth:
target_output = [1, 0, 0]

loss = -(math.log(
    softmax_outputs[0]*target_output[0] +
    softmax_outputs[1]*target_output[1] +
    softmax_outputs[2]*target_output[2]
))

print(loss)
#  The same above code can be simplied by taking in accounts for the 0's multplication and changing the target value with it's correct index.

simplified_loss = - (math.log(softmax_outputs[0]))

# For example:
# If confidence level might look like [0.22, 0.6, 0.18] or [0.32, 0.36, 0.32]. In both cases, the argmax of these vectors will return the second class as the prediction, but the model’s confidence about these predictions is high only for one of them. The Categorical Cross-Entropy Loss accounts for that and outputs a larger loss the lower the confidence is.

""" 
    Log is short for logarithm and is defined as the solution for the x-term in an equation of the form ax = b. For example, 10x = 100 can be solved with a log: log10 (100), which evaluates to 2.
     
    This property of the log function is especially beneficial when e (Euler's number or ~2.71828) is used in the base (where 10 is in the example). The logarithm with e as its base is referred to as the natural logarithm, natural log, or simply log — you may also see this written as ln : ln(x) = log(x) = log e (x). 
    
    The variety of conventions can make this confusing,so to simplify things, any mention of log will always be a natural logarithm throughout this book. The natural log represents the solution for the x-term in the equation ex = b; for example, ex = 5.2 is solved by log(5.2).
"""
