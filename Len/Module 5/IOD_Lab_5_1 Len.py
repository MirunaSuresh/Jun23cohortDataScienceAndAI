#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src=https://www.institutedata.com/wp-content/uploads/2019/10/iod_h_tp_primary_c.svg width="300">
# </div>

# # Lab 5.1 
# # *Logistic Regression*

# ## Predicting Survival on the Titanic
# 
# The Titanic sank during her maiden voyage after colliding with an iceberg (April 15, 1912). Due to a commercial decision there were insufficient lifeboats, a fact that was partially responsible for the loss 1,502 out of 2,224 passengers and crew. 
# 
# The Titanic dataset incorporates many features of typical real-world problems: a mixture of continuous and discrete features, missing data, linear covariance, and an element of random chance. Predicting survival therefore involves many practical data science skills.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# ### 1. Load Data
# 
# #Load the `titanic.csv` file into a DataFrame named "titanic", with index column = `PassengerId`. Display the head of the DataFrame.
# 
# titanic = pd.read_csv('/Users/lenkwok/Downloads/projects/titanic.csv',index_col='PassengerId')

# In[14]:


titanic.head()


# Why would we want to set an index column based on `PassengerId`?

# ANSWER: This column is the key to training and testing our model. We use it to partition the dataset and to test the predictions of our model against known outcomes.

# <a name="datadictionary"></a>
# ### 2. Data Dictionary 
# 
# If a data dictionary is available, it is handy to include it in the notebook for reference:
# 
# | Variable |                                 Definition | Key                                            |
# |----------|-------------------------------------------:|------------------------------------------------|
# | Survival | Survival                                   | 0 = No, 1 = Yes                                |
# | Pclass   | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
# | Sex      | Sex                                        |                                                |
# | Age      | Age in years                               |                                                |
# | SibSp    | # of siblings / spouses aboard the Titanic |                                                |
# | Parch    | # of parents / children aboard the Titanic |                                                |
# | Ticket   | Ticket number                              |                                                |
# | Fare     | Passenger fare                             |                                                |
# | Cabin    | Cabin number                               |                                                |
# | Embarked | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |

# ### 2. EDA
# 
# Explore dataset. Find features to predict `Survived`. Get rid of null values. 

# In[15]:


titanic.shape


# In[17]:


titanic.Survived.value_counts()


# In[25]:


titanic.isnull().sum()


# In[26]:


#Find median age of each sex
titanic.groupby("Sex")["Age"].median()


# In[27]:


titanic.groupby("Sex")["Age"].transform("median")


# In[28]:


#Create a new column called "Age Imp"
titanic["Age_Imp"] = titanic.groupby("Sex")["Age"].transform("median")


# In[31]:


# Fill all NA values in 'age' column with median values and save into 'age imp' column
titanic["Age_Imp"]=titanic["Age"].fillna(titanic.groupby("Sex")["Age"].transform("median"))


# In[34]:


#verify no more null values in 'Age Imp' column
titanic['Age_Imp'].isnull().sum()


# ### 3. Numerical Predictors Only

# #### 3.1. Set Target and Features
# 
# To begin, let's try a model based on the passenger class (`Pclass`) and parents/children features (`Parch`):

# In[37]:


feature_cols=['Pclass',"Parch"]
X=titanic[feature_cols]
y=titanic['Survived']


# #### 3.2 Partition

# Partition the data into training and testing subsets:
# 
# - Use `random_state` = 1

# In[38]:


# ANSWER
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)


# #### 3.3. Build Model
# 
# Prepare a model by creating an instance of the `LogisticRegression` class from the `sklearn.linear_model` library:

# In[40]:


# ANSWER
# Create Model
logreg=LogisticRegression(solver='liblinear',max_iter=10000)


# Now train it on the training data subset, using the `fit` method of the model object (Nb. by default, `fit` will print the hyperparameters of the model):

# In[41]:


# ANSWER
# Fit Model
logreg.fit(X_train,y_train)


# The computed coefficients are an array (`coef_`) stored in the 1st element of an array:

# In[43]:


# ANSWER
logreg.coef_


# The computed intercept (`intercept_`) is the 1st element of another array:

# In[44]:


# ANSWER
logreg.intercept_


# We can create tuples of the predictor names and coefficients like this:

# In[46]:


# ANSWER
print(set(zip(feature_cols, logreg.coef_[0])))


# If we want formatted output, here is a neat way to list the coefficients by predictor:

# In[ ]:


for col in zip(X_train.columns, model.coef_[0]):
    print('{:<10s}  {:+.06f}'.format(col[0], col[1]))  # Nb. increase 10 for longer names


# This result implies that survival declines with passenger class (i.e. 1st class is highest) but increases with the number of parents or children in a group.

# Let's see how well the model fit the training data. The `accuracy_score` is the proportion of correct predictions:

# In[ ]:


# ANSWER


# What is the  `accuracy_score` for the test data?

# In[ ]:


# ANSWER


# What can we say aout this result?

# ANSWER
# - ...
# - ...

# #### 3.4. Add `AGE` as Feature

# Let's include `Age` in the model. As we know from our EDA, this feature has many missing values. We don't want to throw away so many rows, so we will replace `NA` values with imputed values (e.g. the overall mean age):

# In[ ]:


# ANSWER


# In[ ]:


# Build Model

# Fit Model

# Score


# So, including age did little to reduce the variance in our model. Why might this be?

# ANSWER
# 
# - ...
# - ...
# - ...

# Let's see where the model is going wrong by showing the Confusion Matrix:

# In[ ]:


# ANSWER
y_pred_class = logreg.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred_class))


# Nb. Here is how `confusion_matrix` arranges its output:

# In[ ]:


print(np.asarray([['TN', 'FP'], ['FN', 'TP']]))


# Which type of error is more prevalent?

# ANSWER: ...

# Maybe we aren't using the right cut-off value. By default, we are predicting that `Survival` = True if the probability >= 0.5, but we could use a different threshold. The ROC curve helps us decide (as well as showing us how good our predictive model really is):

# In[ ]:


# Generate the prediction values for each of the test observations using predict_proba() function rather than just predict
preds = logreg.predict_proba(X_test)[:,1]

# Store the false positive rate(fpr), true positive rate (tpr) in vectors for use in the graph
fpr, tpr, _ = metrics.roc_curve(y_test, preds)

# Store the Area Under the Curve (AUC) so we can annotate our graph with theis metric
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC Curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc = "lower right")
plt.show()


# ### 4. Including Categorical Predictors

# So far, we've only used numerical features for prediction. Let's convert the character features to dummy variables so we can include them in the model:

# In[ ]:


titanic_with_dummies = pd.get_dummies(data = titanic, columns = ['Sex', 'Embarked', 'Pclass'], 
                                      prefix = ['Sex', 'Embarked', 'Pclass'] )
titanic_with_dummies.head()


# So, this created a column for every possible value of every categorical variable. (A more compact approach would have been to reduce the number of dummy variables by one for each feature, so that the first vriable from each captures two possible states.)

# Now that we have data on sex, embarkation port, and passenger class we can try to improve our `Age` imputation by stratifying it by the means of groups within the passenger population:

# In[ ]:


titanic_with_dummies['Age'] = titanic_with_dummies[["Age", "Parch", "Sex_male", "Pclass_1", "Pclass_2"]].groupby(["Parch", "Sex_male", "Pclass_1", "Pclass_2"])["Age"].transform(lambda x: x.fillna(x.mean()))


# Now train the model using the expanded set of predictors and compute the accuracy score for the test set:

# In[ ]:


# ANSWER
# Set Feature Both Numerical, Categorical


# Plot the ROC curve for the new model:

# In[ ]:


# ANSWER


# Can we improve the model by including the remaining features?

# In[ ]:


# ANSWER


# ## Homework
# 
# 1. Remove the `random_state` parameter (if you have used), so that the data partition will be different every time, and run through the final modelling process a few times. Do the results change?
# 
# 2. Use cross-validation to assess the quality of the model when overfitting is controlled. Does the accuracy improve?
# 
# 3. Look at the `fpr` & `tpr` vectors for the best model.

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
