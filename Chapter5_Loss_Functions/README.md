# Notes for Calculating Network error with Loss Functions:

## ***Quick Intro:***

The loss function is also referred to as the *cost function*. *The loss function is the algorithm that quantifies how wrong the model is. Loss is hence the measure of this metric(the algorithm).* Since loss is a model we want to to be as close to as 0.Recall our earlier example of *confidence*: [0.22, 0.6, 0.18] vs [0.32, 0.36, 0.32]. 

If the correct class were indeed the middle one (index 1), the model accuracy would be identical between the two above. But are these two examples really​ as accurate as each other They are not, because accuracy is simply applying an argmax to the output to find the index of the biggest value.

*The output of a neural network is actually *confidence*, and more confidence in the correct answer is better.*Because of this, we strive to increase correct confidence and decrease misplaced confidence. 

### **1.*Categorical Cross Entropy Loss:***

*Categorical cross-entropy is explicitly used to compare a “ground-truth” probability (y ​ or ​ “targets​”) and some predicted distribution (y-hat or “predictions​”)*, so it makes sense to use cross-entropy here. While *the log loss error is used for Binary Logistic Regression models.*

![Formula for categorical crossentropy-](image.png)

<!-- The formula for calculating the categorical cross-entropy of y(actual/desired distribution) and y-hat (predicted distribution) is: 
            Li =  Negative Summation of Yijlog(y-Hatij) -->

Where Li denotes sample loss value,
i is the i-th sample in the set,
j is the label/output index, 
y denotes the target values, and y-hat denotes the predicted values.


## ***So what is log?***
*Log is short for logarithm and is defined as the solution for the x-term in an equation of the form ax = b.* For example, **10x = 100** can be solved with a **log: log10 (100)**, which evaluates to **2**.
    
This property of the log function is especially beneficial when *e* (Euler's number or ~2.71828) is used in the base (where 10 is in the example). *The logarithm with e as its base is referred to as the natural logarithm,* natural log, or simply log — you may also see this written as ***ln : ln(x) = log(x) = log e (x)*** 

The variety of conventions can make this confusing,so to simplify things, any mention of log ***will always be using a natural logarithm for our implemenatation.*** The natural log represents the solution for the x-term in the equation ex = b; for example, ex = 5.2 is solved by log(5.2).  