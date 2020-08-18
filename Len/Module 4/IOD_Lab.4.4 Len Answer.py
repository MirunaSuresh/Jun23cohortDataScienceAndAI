#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src=https://www.institutedata.com/wp-content/uploads/2019/10/iod_h_tp_primary_c.svg width="300">
# </div>

# # Lab 4.2.2: Regularisation

# In[1]:


## Import Libraries

import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

## Avoid some version change warnings
import warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')


# ### 1. Regularisation
# 
# The goal of "regularizing" regression models is to structurally prevent overfitting by imposing a penalty on the coefficients of the model.
# 
# Regularization methods like the Ridge and Lasso add this additional "penalty" on the size of coefficients to the loss function. When the loss function is minimized, this additional component is added to the residual sum of squares.
# 
# In other words, the minimization becomes a balance between the error between predictions and true values and the size of the coefficients. 
# 
# The two most common types of regularization are the **Lasso**, **Ridge**. 

# #### 1.1 Load Diabetics Data Using datasets of sklearn
# 
# Hint: Check Lab 4.3

# In[2]:


## Load the Diabetes dataset

# Load the diabetes dataset from sklearn
diabetes = datasets.load_diabetes()


# In[3]:


# Description
print(diabetes.DESCR)


# In[7]:


# Predictors
X = pd.DataFrame(diabetes.data, columns = diabetes.feature_names)


# In[8]:


# Target
y = diabetes.target


# #### 1.2 Create a Base Model Using Linear Regression

# In[9]:


# Create Model
model = LinearRegression()


# In[10]:


# Fit
model.fit(X, y)


# In[11]:


# Score
#A model that can give a goodness of fit measure or a likelihood of unseen data, implements (higher is better):

model.score(X, y)


# In[ ]:


# Check Coeffiricent


# In[12]:


def view_coeff(X, model):
    model_coefs = pd.DataFrame({'variable': X.columns,
                                'coef': model.coef_,
                                'abs_coef': np.abs(model.coef_)})
    model_coefs.sort_values('abs_coef', inplace=True, ascending=False)
    sns.barplot(x="variable", y="coef", data=model_coefs)


# In[13]:


# Plot Coefficients
view_coeff(X, model)


# #### 1.3 Ridge
# 
# ##### 1.3.1 Calculate Ridge Regression model

# In[16]:


## Calculate Ridge Regression model

# create a model object to hold the modelling parameters
clf = Ridge()

# keep track of the intermediate results for coefficients and errors
coefs = []
errors = []

# create a range of alphas to calculate
#Return numbers spaced evenly on a log scale.
ridge_alphas = np.logspace(-6, 6, 200)

# Train the model with different regularisation strengths
for a in ridge_alphas:
    clf.set_params(alpha = a)
    clf.fit(X, y)
    coefs.append(clf.coef_)
    errors.append(mean_squared_error(clf.coef_, model.coef_))


# In[18]:


print(ridge_alphas)


# ##### 1.3.2 Visual Represenantion of Coefficient of Ridge Model

# In[17]:


# Display results
plt.figure(figsize = (20, 6))

plt.subplot(121)
ax = plt.gca()
ax.plot(ridge_alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularisation')
plt.axis('tight')

plt.subplot(122)
ax = plt.gca()
ax.plot(ridge_alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Coefficient error as a function of the regularisation')
plt.axis('tight')

plt.show()


# ##### 1.3.3. [BONUS]  Find an optimal value for Ridge regression alpha using `RidgeCV`.
# 
# [Go to the documentation and read how RidgeCV works.](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV).
# 
# Note here that you will be optimizing both the alpha parameter and the l1_ratio:
# - `alpha`: strength of regularization

# In[19]:


optimal_ridge = RidgeCV(alphas=ridge_alphas, cv=10)
optimal_ridge.fit(X, y)
print('Alpha:', optimal_ridge.alpha_)
print('Score:', optimal_ridge.score(X, y))


# In[20]:


view_coeff(X, optimal_ridge)


# In[21]:


optimal_ridge = RidgeCV(alphas=ridge_alphas, cv=20)
optimal_ridge.fit(X, y)
print('Alpha:', optimal_ridge.alpha_)
print('Score:', optimal_ridge.score(X, y))


# In[22]:


view_coeff(X, optimal_ridge)


# #### 1.4 Lasso
# 
# ##### 1.4.1 Calculate Lasso Regression model

# In[23]:


## Calculate Lasso Regression model

# create a model object to hold the modelling parameters
clf = Lasso()

# keep track of the intermediate results for coefficients and errors
coefs = []
errors = []

# create a range of alphas to calculate
lasso_alphas = np.logspace(-6, 6, 200)

# Train the model with different regularisation strengths
for a in lasso_alphas:
    clf.set_params(alpha = a)
    clf.fit(X, y)
    coefs.append(clf.coef_)
    errors.append(mean_squared_error(clf.coef_, model.coef_))


# ##### 1.4.2 Visual Represenantion of Coefficient of Lasso Model
# 
# Hint: Same as Ridge

# In[24]:


# Display results
plt.figure(figsize = (20, 6))

plt.subplot(121)
ax = plt.gca()
ax.plot(lasso_alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularisation')
plt.axis('tight')

plt.subplot(122)
ax = plt.gca()
ax.plot(lasso_alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Coefficient error as a function of the regularisation')
plt.axis('tight')

plt.show()


# ##### 1.4.3. [BONUS]  Find an optimal value for Loass regression alpha using `LassoCV`.
# 
# [Go to the documentation and read how LassoCV works.](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV).
# 
# Note here that you will be optimizing both the alpha parameter and the l1_ratio:
# - `alpha`: strength of regularization

# In[25]:


optimal_lasso = LassoCV(alphas=lasso_alphas, cv=10)
optimal_lasso.fit(X, y)
print('Alpha:', optimal_lasso.alpha_)
print('Score:', optimal_lasso.score(X, y))


# In[26]:


# Plot Coefficient
view_coeff(X, optimal_lasso)


# ### 2. [Bonus] Compare the residuals for the Ridge and Lasso visually.
# 
# Find and create sctterplot for both Ridge and Lasso residuals.

# In[28]:


# Build the ridge and lasso using optimal alpha

ridge = Ridge(alpha=optimal_ridge.alpha_)
lasso = Lasso(alpha=optimal_lasso.alpha_)

# Need to fit the Lasso and Ridge outside of cross_val_score like we did with the ridge
ridge.fit(X, y)
lasso.fit(X, y)


# In[29]:


# model residuals:
ridge_resid = y - ridge.predict(X)
lasso_resid = y - lasso.predict(X)


# In[30]:


# Jointplot
sns.jointplot(ridge_resid, lasso_resid);

