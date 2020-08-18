#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src=https://www.institutedata.com/wp-content/uploads/2019/10/iod_h_tp_primary_c.svg width="300">
# </div>

# ## Lab 4.2.1: Feature Selection

# ### 1. Load & Explore Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# #### 1.1 Load Data

# In[2]:


# Read CSV
wine = pd.read_csv('/Users/lenkwok/Desktop/projects/winequality_merged.csv')


# #### 1.2 Explore Data (Exploratory Data Analysis)

# In[3]:


wine.head()


# In[4]:


wine.shape


# In[5]:


wine.isnull().sum()


# In[6]:


wine.info()


# In[16]:


# Correlation
wine_corr = wine.corr()
wine_corr


# In[7]:


sns.pairplot(wine)
#too small to read, try again using seaborn


# In[17]:


# Copied code from seaborn examples
# https://seaborn.pydata.org/examples/many_pairwise_correlations.html
sns.set(style="white")

# Generate a mask for the upper triangle
mask = np.zeros_like(wine_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(wine_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.show()


# ### 2. Set Target Variable

# Create a target variable for wine quality.

# In[8]:


y=wine['quality']


# ### 3. Set Predictor Variables

# Create a predictor matrix with variables of your choice. State your reason.

# In[9]:


# use correlation function
wine.corr()['quality'].sort_values()


# In[10]:


#choose density, volatile acidity, chloreides, alchohol
predictor_columns=['density','volatile acidity','chlorides','alcohol']


# In[11]:


X=wine[predictor_columns]
X.head()


# ### 4. Using Linear Regression Create a Model and Test Score

# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[20]:


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[14]:


# Create a model for Linear Regression

# Fit the model with the Training data

# Calculate the score (R^2 for Regression) for Training Data

# Calculate the score (R^2 for Regression) for Testing Data


# In[24]:


regressor=LinearRegression()
regressor.fit(X_train,y_train)
linreg.score(X_train,y_train)


# In[23]:


# Fit and score model on training data

linreg = LinearRegression()
linreg.fit(X_train,y_train)

linreg.score(X_train,y_train)


# In[26]:


# Score model on test data

linreg.score(X_test, y_test)


# In[28]:


# Find coefficients

linreg_coef_df = pd.DataFrame({'feature': X.columns, 'coefficient': linreg.coef_})
linreg_coef_df


# # BONUS: Cross validation

# In[29]:


# Cross validation 
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error


# In[33]:


# Set up 5-fold cross validation  
k_fold = KFold(5, shuffle=True)
train_scores = []
train_rmse = []
test_scores = []
test_rmse = []

for k, (train, test) in enumerate(k_fold.split(X)):
    
    # Get training and test sets for X and y
    X_train = X.iloc[train, ]
    y_train = y.iloc[train, ]
    X_test = X.iloc[test, ]
    y_test = y.iloc[test, ]
    
    # Fit model with training set
    linreg.fit(X_train, y_train)
    
    # Make predictions with training and test set
    train_preds = linreg.predict(X_train)
    test_preds = linreg.predict(X_test)
    
    # Score R2 and RMSE on training and test sets and store in list
    train_scores.append(linreg.score(X_train, y_train))
    test_scores.append(linreg.score(X_test, y_test))
    
    train_rmse.append(mean_squared_error(y_train, train_preds, squared=False))
    test_rmse.append(mean_squared_error(y_test, test_preds, squared=False))

# Create a metrics_df dataframe to display r2 and rmse scores
metrics_df = pd.DataFrame({'Training R2': train_scores, 
                           'Test R2': test_scores, 
                           'Training RMSE': train_rmse, 
                           'Test RMSE': test_rmse},
                          index=[i+1 for i in range(5)])

metrics_df


# In[34]:


metrics_df.describe()


# **Please continue with Lab 4.2.2 with the same dataset.**
