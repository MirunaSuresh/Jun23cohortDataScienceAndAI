#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src=https://www.institutedata.com/wp-content/uploads/2019/10/iod_h_tp_primary_c.svg width="300">
# </div>

# # Lab 5.3 
# # *The Perceptron*

# In[ ]:


# Personal summary: 

#Bias of -1 introduced into X array. SGD needs this.  Probably for matrix calculations? convert 2D to 3D matrix?
#In code block 9, SGD algorithm is run with loop iterations for number of epochs and also number of samples (using enumerate).  
# If dot product is <=0, there is misclassfication. Update weights.  If dot product is >0, no need.
#For epochs = 10, plot shows misclassfication of points.   Gradient of 0 is not achieved.  
#Repeating for epochs = 30 shows gradient went to zero just before 15.  
#Run code for epoch = 15 gives vector of (2,3) and bias of 13.
#np.dot(X[i], w) at the end verifies that our training data, that is X array, gets classified correctly (sign is tve or -ve)


# The perceptron is the basic unit of a neural network. It learns by adjusting the weights applied to each of its inputs until the error at its output is minimised.
# 
# The example in this lab uses the stochastic gradient descent (SGD) algorithm to optimise the weights of a perceptron applied to a 2D classification problem.

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# The training dataset has 2 numeric features (X is 2D) and a binary response (y = +/-1):

# In[2]:


X = np.array([[-2, 4], [4, 1], [1, 6], [2, 4], [6, 2]])
y = np.array([-1, -1, 1, 1, 1])


# Here is the training data, along with a candidate hyperplane that separates the classes:

# In[3]:


def plotData(X):
    for d, sample in enumerate(X):
        # Plot the negative samples
        if d < 2:
            plt.scatter(sample[0], sample[1], s = 120, marker = '_', color = 'blue', linewidths = 2)
        # Plot the positive samples
        else:
            plt.scatter(sample[0], sample[1], s = 120, marker = '+', color = 'blue', linewidths = 2)
    plt.xlabel('X0')
    plt.ylabel('X1')

plotData(X)

# Print a possible hyperplane, that is seperating the two classes:
plt.plot([-2, 6], [6, 0.5], color = 'orange', linestyle = 'dashed')


# The activation function is based on the dot product of 

# 
# 
# ```
# # This is formatted as code
# ```
# 
# We need to include a bias term (-1) in the X array. This will transform the decision boundary so that the sign of the dot product of any data point with the weights vector (represented by ⟨x[i], w⟩ in code commments, below) will determine class membership: 

# In[4]:


X = np.array([ [-2, 4, -1], [4, 1, -1], [1, 6, -1], [2, 4, -1], [6, 2, -1]])


# Here is a simple implementation of the stochastic gradient descent algorithm for computing the weights:

# In[9]:


def perceptron_sgd(Xt, Yt, eta = 1, epochs = 20):
    
    # Initialize the weight vector for the perceptron with zeros:
    wt = np.zeros(len(Xt[0]))
    print(wt)
    for t in range(epochs):
        print('epoch')
        print(t)
        # Iterate over each sample in the data set:
        for i, x in enumerate(Xt):
            print(i)
            print(x)
           
            print(np.dot(Xt[i], wt) * Yt[i])
            # Test for misclassification: y * ⟨x[i], w⟩ <= 0:
            if (np.dot(Xt[i], wt) * Yt[i]) <= 0:
                print('update')
                print(Xt[i]*Yt[i])
                # Update weights:
                wt = wt + eta * Xt[i] * Yt[i]

    return wt


# Compute the weights using default learning rate (eta = 1) and number of epochs = 10:

# In[7]:


w = perceptron_sgd(X, y, epochs = 10)
print(w)


# Did it work? Let's check the decision boundary (hyperplane) and try some predictions:

# In[8]:


def plotHyperplane(wt):
    
    # Nb. Plotting the hyperplance uses some complex tricks ...
    
    x2 = [wt[0], wt[1], -wt[1], wt[0]]
    x3 = [wt[0], wt[1], wt[1], -wt[0]]
    x2x3 = np.array([x2, x3])
    # print(x2x3)
    Xp, yp, U, V = zip(*x2x3)
    # print(Xp, yp, U, V)
    ax = plt.gca()
    ax.quiver(Xp, yp, U, V, scale = 1, color = 'orange')
    
plotData(X)
plotHyperplane(w)

# Test samples:
plt.scatter(2, 2, s = 120, marker = '_', linewidths = 2, color = 'red')
plt.scatter(4, 3, s = 120, marker = '+', linewidths = 2, color = 'red')    


# So, not only is one of the new test points misclassified, one of the training points is also misclassified! 
# 
# Let's a look at how the model training actually proceeds. The error at each epoch is calculated using a hinge-loss function:

# In[13]:


def perceptron_sgd_plot(Xt, Yt, eta = 1, epochs = 10):

    wt = np.zeros(len(Xt[0]))
    errors = []

    for t in range(epochs):
        total_error = 0
        for i, x in enumerate(Xt):
            if (np.dot(Xt[i], wt) * Yt[i]) <= 0:
                total_error += (np.dot(Xt[i], wt) * Yt[i])
                wt += eta * Xt[i] * Yt[i]
        errors.append(total_error * (-1))
        
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    
    return wt

print(perceptron_sgd_plot(X, y))


# So, 10 epochs clearly wasn't enough for the SGD algorithm to converge. 
# 
# Try a increasing `epochs` until the error goes to zero, then replot the test data and decision boundary:

# In[ ]:


#ANSWER
print(perceptron_sgd_plot(X, y, epochs = 30))


# In[ ]:


#ANSWER
plotData(X)
w = perceptron_sgd(X, y, epochs = 15)
print(w)
plotHyperplane(w)

# Test samples:
plt.scatter(2, 2, s = 120, marker = '_', linewidths = 2, color = 'red')
plt.scatter(4, 3, s = 120, marker = '+', linewidths = 2, color = 'red')   


# Show how to manually compute class membership for a new data point Xi = [3.5, 3.3] using just the weights determined above:

# In[ ]:


#ANSWER
def classType(Xi, wt):
    Xi.append(-1)
    test = np.dot(Xi, wt)
    if (test) > 0:
        print('class "+" predicted')
    elif (test) < 0:
        print('class "-" predicted')
    else:
        print('edge case: class "+" predicted')
        
Xi = [3.5, 3.3]
classType(Xi, w)


# In[ ]:


# TEST: make sure the training data get correctly classified:
for i, x in enumerate(X):
    print(X[i], np.dot(X[i], w))


# # Manual calculations for new weights (2,3) with bias -13.
# 1st sample (-2,4) supposed to be -ve:  -2*2 + 4*3-13 = sign(-5) = -
# 2nd sample (4,1) supposed to be -ve:  4*2 + 1*3-13 = sign (-2) = -
# 3rd sample (1,6) supposed to be +ve:  sign(7) = +
# 4th sample (2,4) supposed to be +ve:  sign(3) = +
# Last sample (6, 2) sign (5) = +

# ## === End ===

# >

# >

# >

# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# > > > > > > > > > © 2019 Institute of Data
# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# 
