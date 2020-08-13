#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src=https://www.institutedata.com/wp-content/uploads/2019/10/iod_h_tp_primary_c.svg width="300">
# </div>

# ## Lab 4.2.2: Feature Selection

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
pd.options.display.float_format = '{:.5f}'.format

get_ipython().run_line_magic('matplotlib', 'inline')


# ### 1. Forward Feature Selection
# 
# > Forward Selection: Forward selection is an iterative method in which we start with having no feature in the model. In each iteration, we keep adding the feature which best improves our model till an addition of a new variable does not improve the performance of the model.
# 
# Create a Regression model using Forward Feature Selection by looping over all the features adding one at a time until there are no improvements on the prediction metric ( R2  and  AdjustedR2  in this case).

# #### 1.1 Load Wine Data & Define Predictor and Target

# In[2]:


## Load the wine quality dataset

# Load the wine dataset from csv
wine = pd.read_csv('/Users/lenkwok/Desktop/projects/winequality_merged.csv')
wine.info()


# # Exploratory data analysis
# 

# In[64]:


wine_kor = wine.corr()


# In[61]:


# Copied code from seaborn examples
# https://seaborn.pydata.org/examples/many_pairwise_correlations.html
sns.set(style="white")

# Generate a mask for the upper triangle
mask = np.zeros_like(wine_kor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(wine_kor, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.show()


# In[21]:


sns.pairplot(wine) 


# In[66]:


wine.corr()['quality'].sort_values()


# In[5]:


# define the target variable (dependent variable) as y
y = wine['quality']


X=wine.loc[:, wine.columns != 'quality']


# In[6]:


## Create training and testing subsets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# #### 1.2 Overview of the code below
# 
# The external `while` loop goes forever until there are no improvements to the model, which is controlled by the flag `changed` (until is **not** changed).
# The inner `for` loop goes over each of the features not yet included in the model and calculates the correlation coefficient. If any model improves on the previous best model then the records are updated.
# 
# #### Code variables
# - `included`: list of the features (predictors) that were included in the model; starts empty.
# - `excluded`: list of features that have **not** been included in the model; starts as the full list of features.
# - `best`: dictionary to keep record of the best model found at any stage; starts 'empty'.
# - `model`: object of class LinearRegression, with default values for all parameters.
# 
# #### Methods of the `LinearRegression` object to investigate
# - `fit()`
# - `fit.score()`
# 
# #### Adjusted $R^2$ formula
# $$Adjusted \; R^2 = 1 - { (1 - R^2) (n - 1)  \over n - k - 1 }$$
# 
# #### Linear Regression [reference](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)

# In[9]:


## Flag intermediate output

show_steps = True   # for testing/debugging
# show_steps = False  # without showing steps


# In[10]:


X_train.shape[0]


# In[11]:


# Use Forward Feature Selection to pick a good model

# start with no predictors
included = []
# keep track of model and parameters
best = {'feature': '', 'r2': 0, 'a_r2': 0}
# create a model object to hold the modelling parameters
model = LinearRegression()
# get the number of cases in the test data
n = X_test.shape[0]

while True:
    changed = False
    
    if show_steps:
        print('') 

    # list the features to be evaluated
    excluded = list(set(X.columns) - set(included))
    
    if show_steps:
        print('(Step) Excluded = %s' % ', '.join(excluded))  

    # for each remaining feature to be evaluated
    for new_column in excluded:
        
        if show_steps:
            print('(Step) Trying %s...' % new_column)
            print('(Step) - Features = %s' % ', '.join(included + [new_column]))

        # fit the model with the Training data
        fit = model.fit(X_train[included + [new_column]], y_train)
        # calculate the score (R^2 for Regression)
        r2 = fit.score(X_train[included + [new_column]], y_train)
        # number of predictors in this model
        k = len(included + [new_column])
        # calculate the adjusted R^2
        adjusted_r2 = 1 - ( ( (1 - r2) * (n - 1) ) / (n - k - 1) )

        if show_steps:
            print('(Step) - Adjusted R^2: This = %.3f; Best = %.3f' % 
                  (adjusted_r2, best['a_r2']))

        # if model improves
        if adjusted_r2 > best['a_r2']:
            # record new parameters
            best = {'feature': new_column, 'r2': r2, 'a_r2': adjusted_r2}
            # flag that found a better model
            changed = True
            if show_steps:
                print('(Step) - New Best!   : Feature = %s; R^2 = %.3f; Adjusted R^2 = %.3f' % 
                      (best['feature'], best['r2'], best['a_r2']))
    # END for

    # if found a better model after testing all remaining features
    if changed:
        # update control details
        included.append(best['feature'])
        excluded = list(set(excluded) - set(best['feature']))
        print('Added feature %-4s with R^2 = %.3f and adjusted R^2 = %.3f' % 
              (best['feature'], best['r2'], best['a_r2']))
    else:
        # terminate if no better model
        print('*'*50)
        break

print('')
print('Resulting features:')
print(', '.join(included))


# # Or we can also use sm library

# In[12]:


X_train = sm.add_constant(X_train)
results = sm.OLS(y_train, X_train).fit()
results.summary() # we can remove citric acid as p value>0.05


# In[13]:


X2 = wine.drop(["quality", "citric acid"], axis=1)


# In[14]:


X2.info()


# In[15]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size = 0.2,random_state = 1)


# In[16]:


X2_train = sm.add_constant(X2_train)

results2 = sm.OLS(y2_train, X2_train).fit()
results2.summary()
#chlorides has P-value just a bit higher than 0.05


# Quality = 99.3098 + 0.0684*fixed acidity -1.4900 * volatile acidity +-0.6920	*residual sugar        
# + 0.0049* free sulfur dioxide   
#  -0.0014* total sulfur dioxide  
#  -98.5414* density              
#  +0.4062*pH                   
# +0.7153	*sulphates            
# + 0.2361	* alcohol            
#  + 0.3741* red_wine
#  
#  #It seems that quality is negatively affected by density most of all.
