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

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')


# ### 1. Load Data
# 
# Load the `titanic.csv` file into a DataFrame named "titanic", with index column = `PassengerId`. Display the head of the DataFrame.

# In[3]:


# ANSWER
titanic = pd.read_csv('/Users/lenkwok/Downloads/projects/titanic.csv', index_col='PassengerId')


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

# In[3]:


# Shape
titanic.shape


# In[4]:


# Head
titanic.head()


# In[5]:


# Check how many data are missing in these columns
titanic.isnull().sum()


# In[12]:


def facetgridplot(train, var):
    facet = sns.FacetGrid(train, hue="Survived", aspect=4)
    facet.map(sns.kdeplot, var, shade= True)
    facet.set(xlim=(0, train[var].max()))
    facet.add_legend()
    plt.show();


# In[14]:


def bar_chart(train, feature):
    survived = train[train['Survived']==1][feature].value_counts(normalize=True)*100
    dead = train[train['Survived']==0][feature].value_counts(normalize=True)*100
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[8]:


# Sex
bar_chart(titanic, 'Sex')


# In[9]:


# Pclass
bar_chart(titanic, 'Pclass')


# In[10]:


# Embarked
bar_chart(titanic, 'Embarked')


# In[8]:


# Find median age by sex

titanic.groupby("Sex")["Age"].median()


# In[9]:


# The following is an array of the median age by sex
titanic.groupby("Sex")["Age"].transform("median")


# In[35]:


# fill missing age with median age for each sex (0 (male), 1 (female))
titanic["Age"].fillna(titanic.groupby("Sex")["Age"].transform("median"))


# In[36]:


# Age
facetgridplot(titanic, 'Age')


# In[37]:


# Fare
facetgridplot(titanic, 'Fare')


# In[16]:


# fill missing embarked with `S` as most people embarked from there
titanic['Embarked'].fillna('S', inplace=True)


# In[18]:


bar_chart(titanic, 'Embarked')


# ### 3. Numerical Predictors Only

# #### 3.1. Set Target and Features
# 
# To begin, let's try a model based on the passenger class and parents/children features:

# In[38]:


# ANSWER
feature_cols = ['Pclass', 'Parch']
X = titanic[feature_cols]
y = titanic['Survived']


# #### 3.2 Partition

# Partition the data into training and testing subsets:

# In[39]:


# ANSWER
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# #### 3.3. Build Model
# 
# Prepare a model by creating an instance of the `LogisticRegression` class from the `sklearn.linear_model` library:

# In[40]:


# ANSWER
# Create Model
logreg = LogisticRegression()


# Now train it on the training data subset, using the `fit` method of the model object (Nb. by default, `fit` will print the hyperparameters of the model):

# In[41]:


# ANSWER
# Fit Model
logreg.fit(X_train, y_train)


# The computed coefficients are an array stored in the 1st element of an array:

# In[42]:


# ANSWER
logreg.coef_


# The computed intercept is the 1st element of another array:

# In[25]:


# ANSWER
logreg.intercept_


# We can create tuples of the predictor names and coefficients like this:

# In[43]:


# ANSWER
print(set(zip(feature_cols, logreg.coef_[0])))


# If we want formatted output, here is a neat way to list the coefficients by predictor:

# In[44]:


for col in zip(X_train.columns, logreg.coef_[0]):
    print('{:<10s}  {:+.06f}'.format(col[0], col[1]))  # Nb. increase 10 for longer names


# This result implies that survival declines with passenger class (i.e. 1st class is highest) but increases with the number of parents or children in a group.

# Let's see how well the model fit the training data. The `accuracy_score` is the proportion of correct predictions:

# In[45]:


print('accuracy = {:7.4f}'.format(logreg.score(X_train, y_train)))


# What is the  `accuracy_score` for the test data?

# In[47]:


#?
print('accuracy = {:7.4f}'.format(logreg.score(X_test, y_test)))


# What can we say aout this result?

# ANSWER
# - test set is predicted almost as well as training set
# - overfitting seems unlikely

# #### 3.4. Add `AGE` as Feature

# Let's include `Age` in the model. As we know from our EDA, this feature has many missing values. We don't want to throw away so many rows, so we will replace `NA` values with imputed values (e.g. the overall mean age):

# In[48]:


# ANSWER
titanic['Age'].fillna(titanic.Age.mean(), inplace=True)
feature_cols = ['Pclass', 'Parch', 'Age']
X = titanic[feature_cols]


# In[49]:


# Build Model
logreg = LogisticRegression()

# Fit Model
logreg.fit(X_train, y_train)

# Score
logreg_score = logreg.score(X_train, y_train)
print('accuracy = {:1.4f}'.format(logreg_score))


# So, including age did little to reduce the variance in our model. Why might this be?

# ANSWER
# 
# - age is not related to Titanic survival
# - age is not independent of other features already in the model
# - imputing the missing values distorted the distribution too much

# Let's see where the model is going wrong by showing the Confusion Matrix:

# In[50]:


y_pred_class = logreg.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred_class))


# Nb. Here is how `confusion_matrix` arranges its output:

# In[33]:


print(np.asarray([['TN', 'FP'], ['FN', 'TP']]))


# Which type of error is more prevalent?

# ANSWER:Type 2 (false negatives).

# Maybe we aren't using the right cut-off value. By default, we are predicting that `Survival` = True if the probability >= 0.5, but we could use a different threshold. The ROC curve helps us decide (as well as showing us how good our predictive model really is):

# In[51]:


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


# In[52]:


fpr


# ### 4. Including Categorical Predictors

# So far, we've only used numerical features for prediction. Let's convert the character features to dummy variables so we can include them in the model:

# In[53]:


titanic_with_dummies = pd.get_dummies(data = titanic, columns = ['Sex', 'Embarked', 'Pclass'], 
                                      prefix = ['Sex', 'Embarked', 'Pclass'] )
titanic_with_dummies.head()


# So, this created a column for every possible value of every categorical variable. (A more compact approach would have been to reduce the number of dummy variables by one for each feature, so that the first vriable from each captures two possible states.)

# Now that we have data on sex, embarkation port, and passenger class we can try to improve our `Age` imputation by stratifying it by the means of groups within the passenger population:

# In[54]:


titanic_with_dummies['Age'] = titanic_with_dummies[["Age", "Parch", "Sex_male", "Pclass_1", "Pclass_2"]].groupby(["Parch", "Sex_male", "Pclass_1", "Pclass_2"])["Age"].transform(lambda x: x.fillna(x.mean()))


# In[55]:


titanic_with_dummies[["Age", "Parch", "Sex_male", "Pclass_1", "Pclass_2"]].groupby(["Parch", "Sex_male", "Pclass_1", "Pclass_2"])["Age"].mean()


# In[42]:


titanic_with_dummies['Age'].value_counts()


# In[56]:


titanic_with_dummies


# Now train the model using the expanded set of predictors and compute the accuracy score for the test set:

# In[57]:


def get_logreg_score(data, feature_cols, target_col):
    X = data[feature_cols]
    y = data[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

    # Build Model
    logreg = LogisticRegression()

    # Fit
    logreg.fit(X_train, y_train)

    # Score
    logreg_score = logreg.score(X_test, y_test)

    # Return accuracy rate
    return logreg_score


# In[58]:


# ANSWER
# Set Feature Both Numerical, Categorical
target_col = 'Survived'
feature_cols = ['Parch', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Age', 'SibSp']
get_logreg_score(titanic_with_dummies, feature_cols, target_col)


# Plot the ROC curve for the new model:

# In[59]:


# ANSWER
def plot_roc_curve(X_test, y_test):
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


# In[ ]:


# Train
plot_roc_curve(X_train, y_train)


# In[60]:


# Test
plot_roc_curve(X_test, y_test)


# Can we improve the model by including the remaining features?

# In[61]:


# ANSWER 
target_col = 'Survived'
feature_cols = ['Age', 'SibSp', 'Parch', 'Fare', 
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'Pclass_1', 'Pclass_2', 'Pclass_3']
get_logreg_score(titanic_with_dummies, feature_cols, target_col)


# In[62]:


ttwd = titanic_with_dummies
print("Male survival: {:5.2f}% of {}\nFemale survival: {:5.2f}% of {}\nChild survival: {:5.2f}% of {}".format(
                                            100 * ttwd[ttwd.Sex_male == 1].Survived.mean(), ttwd.Sex_male.sum(), 
                                            100 * ttwd[ttwd.Sex_female == 1].Survived.mean(), ttwd.Sex_female.sum(),
                                            100 * ttwd[ttwd.Age < 16].Survived.mean(), len(ttwd[ttwd.Age < 16])))


# In[63]:


X = titanic_with_dummies[feature_cols]
y = titanic_with_dummies[target_col]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

# Build Model
logreg = LogisticRegression()

# Fit
logreg.fit(X_train, y_train)

# Make a data frame of results, including actual and predicted response:
y_hat = logreg.predict(X)
y_hats = pd.Series(y_hat, index = X.index)
y_hatdf = pd.DataFrame({'y_hat': y_hats})
y_actuals = pd.Series(y, index = X.index)    # names 'y', 'y_test' are in use
y_actualdf = pd.DataFrame({'y_actual': y_actuals})   
ttall = y_hatdf.join(y_actualdf).join(X)
ttall.head()


# In[64]:


print("Male survival: {:5.2f}% of {}\nFemale survival: {:5.2f}% of {}\nChild survival: {:5.2f}% of {}".format(
                                                    100 * ttall[ttall.Sex_male == 1].y_hat.mean(), ttall.Sex_male.sum(), 
                                                    100 * ttall[ttall.Sex_female == 1].y_hat.mean(), ttall.Sex_female.sum(),
                                                    100 * ttall[ttall.Age < 16].y_hat.mean(), len(ttall[ttall.Age < 16])))


# ## Homework
# 
# 1. Remove the `random_state` parameter, so that the data partition will be different every time, and run through the final modelling process a few times. Do the results change?
# 
# 2. Use cross-validation to assess the quality of the model when overfitting is controlled. Does the accuracy improve?
# 
# 3. Look at the `fpr` & `tpr` vectors for the best model.

# #### 2. Use Cross-Validation
# 
# Use cross-validation to assess the quality of the model when overfitting is controlled. Does the accuracy improve?

# In[65]:


from sklearn.model_selection import cross_val_score
target_col = 'Survived'
feature_cols = ['Age', 'SibSp', 'Parch', 'Fare', 
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'Pclass_1', 'Pclass_2', 'Pclass_3']
logreg = LogisticRegression()
scores = cross_val_score(logreg, titanic_with_dummies[feature_cols], titanic_with_dummies[target_col], cv=5)
scores.mean()


# #### 3. Look at the fpr & tpr vectors for the best model.

# In[67]:


# Build Model
logreg = LogisticRegression()

# Fit
logreg.fit(X_train, y_train)

# Predict
y_pred_class = logreg.predict(X_test)

# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)

#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]


# **Classification Accuracy:** Overall, how often is the classifier correct?

# In[68]:


# use float to perform true division, not integer division
print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, y_pred_class))


# **Classification Error:** Overall, how often is the classifier incorrect?
# 
# Also known as "Misclassification Rate"

# In[ ]:


#"Misclassification Rate"
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print(classification_error)
print(1 - metrics.accuracy_score(y_test, y_pred_class))


# **Sensitivity:** When the actual value is positive, how often is the prediction correct?
# 
# Something we want to maximize
# How "sensitive" is the classifier to detecting positive instances?
# - Also known as "True Positive Rate" or "Recall"
# - TP / all positive
#     - all positive = TP + FN

# In[ ]:


#True Positive Rate" or 
#"Recall"

sensitivity = TP / float(FN + TP)
print(sensitivity)
print(metrics.recall_score(y_test, y_pred_class))


# **Specificity:** When the actual value is negative, how often is the prediction correct?
# 
# Something we want to maximize
# How "specific" (or "selective") is the classifier in predicting positive instances?
# TN / all negative
# all negative = TN + FP

# In[ ]:


specificity = TN / (TN + FP)

print(specificity)


# **False Positive Rate:** When the actual value is negative, how often is the prediction incorrect?

# In[ ]:


false_positive_rate = FP / float(TN + FP)

print(false_positive_rate)
print(1 - specificity)


# **Precision:** When a positive value is predicted, how often is the prediction correct?
# 
# How "precise" is the classifier when predicting positive instances?

# In[ ]:


precision = TP / float(TP + FP)

print(precision)
print(metrics.precision_score(y_test, y_pred_class))


# # Precision and recall are trade-offs ie maximize one at expense of the other

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
