#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

from matplotlib import pyplot

from collections import Counter

from ipywidgets import *
from IPython.display import display

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[60]:


employee_df=pd.read_csv('/Users/lenkwok/Desktop/Human_Resources_new.csv')


# In[61]:


employee_df.info()


# In[62]:


#Replace the 'Attritition' with binary values before performing any visualizations 
employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[63]:


# What is the current attrition rate of employees?
# 16% of employees left company while 84% stayed, so data is not balanced.
employee_df.Attrition.value_counts(normalize=True)


# In[64]:


employee_df.describe()


# In[65]:


#Check for unique values in each column


# In[66]:


for col in employee_df.columns:
    print(col,":",len(employee_df[col].unique()))


# # Dealing with missing values

# In[67]:


employee_df.isnull().sum()


# In[68]:


# Create a mask to find all rows with null values in PerformanceRating column

null_mask = employee_df['PerformanceRating'].isnull()
employee_df[null_mask].head(5)


# In[69]:


#What kind of values are there in PerformanceRating column?
pr=employee_df['PerformanceRating']


# In[70]:


pr


# In[71]:


# Create a mask to find rows with null values in StandardHours column

null_mask = employee_df['WorkLifeBalance'].isnull()
employee_df[null_mask].head(5)


# In[72]:


#What kind of values are there in WorkLifeBalance column?
wb=employee_df['WorkLifeBalance']


# In[73]:


wb


# In[74]:


#Looks like the missing values come from surveys with numerical ratings.
#Replace missing values with most frequent


# In[75]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
employee_df.iloc[:,:] = imputer.fit_transform(employee_df)


# In[76]:


employee_df.isnull().sum()


# # EDA

# In[77]:


plt.figure(figsize=(8,6))
Attrition=employee_df.Attrition.value_counts()
sns.barplot(x=Attrition.index ,y=Attrition.values)
plt.title('Employee Attrition Overview')
plt.xlabel('Employee Turnover', fontsize=16)
plt.ylabel('Count', fontsize=16)


# In[78]:


employee_df.hist(bins = 30, figsize = (20,20))
# Several features such as 'MonthlyIncome' and 'TotalWorkingYears' are positively skewed.
# It makes sense to drop 'EmployeeCount' and 'Standardhours' since they do not change from one employee to the other.
# We could drop employee number as this is not needed


# In[79]:


#Plot correlation matrix
#but hard to see with this Seaborn correlation matrix
correlations = employee_df.corr()
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True)


# In[80]:


# Correlation code from IOD course
employee_df_corr = employee_df.corr()
employee_df_corr

# Copied code from seaborn examples
# https://seaborn.pydata.org/examples/many_pairwise_correlations.html
sns.set(style="white")

# Generate a mask for the upper triangle
mask = np.zeros_like(employee_df_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(employee_df_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.show()

# Total working years is strongly correlated with Job level, Monthly Income & Age
# Years with current director is strongly related to years at company
#Age has strong corrrelatio with the most variables eg eduction, job level


# In[81]:


# Let's drop 'EmployeeCount' , 'Standardhours' since they seem to be constant
employee_df.drop(['EmployeeCount','StandardHours'], axis=1, inplace=True)


# In[82]:


def facetgridplot(employee_df, var):
    facet = sns.FacetGrid(employee_df, hue="Attrition", aspect=4)
    facet.map(sns.kdeplot, var, shade= True)
    facet.set(xlim=(0, employee_df[var].max()))
    facet.add_legend()
    plt.show();


# In[83]:


facetgridplot(employee_df, 'Age')


# In[84]:


facetgridplot(employee_df, 'DistanceFromHome')


# In[85]:


facetgridplot(employee_df,'HourlyRate')


# In[86]:


facetgridplot(employee_df,'EnvironmentSatisfaction')


# In[87]:


plt.figure(figsize=(10, 5))

sns.boxplot(x = 'MonthlyIncome', y = 'Gender', data = employee_df)


# In[88]:


plt.figure(figsize=(10, 5))

sns.boxplot(x = 'TotalWorkingYears', y = 'Gender', data = employee_df)


# In[89]:


plt.figure(figsize=(15, 10))
sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data = employee_df)


# In[90]:


plt.figure(figsize=(25, 5))

sns.countplot(x = 'JobRole', hue = 'Attrition', data = employee_df)


# In[91]:


plt.figure(figsize=(20, 5))

sns.countplot(x = 'MaritalStatus', hue = 'Attrition', data = employee_df)


# In[92]:


plt.figure(figsize=(20, 5))

sns.countplot(x = 'JobInvolvement', hue = 'Attrition', data = employee_df)


# In[93]:


plt.figure(figsize=(20, 5))

sns.countplot(x = 'JobLevel', hue = 'Attrition', data = employee_df)


# In[94]:


#Attrition is very high with employees having age in between 18 to mid 30s. 
#Attrition is more when the distance of office is more from home
#Attrition is highest when pay is in the middle, reflecting heavier responsibilities.  At highest rates of pay, nobody wants to leave
#Attrition is mixed for environmental satisfaction.
#Sales roles and lab technicians suffer the greatest attrition rates.
#Singles leave more often than married or divorced.
#Lower job involvment and levels equals higher chance of leaving.


# # Dealing with categorical values

# In[95]:


employee_df.info()


# In[96]:


#Let's group category values in training data under X_cat
X_cat=employee_df[['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']]


# In[97]:


X_cat


# In[98]:


#Need to drop NAN values before One Hot Encoding for category values
X_cat.dropna(subset=['MaritalStatus'],inplace=True)


# In[99]:


X_cat.isnull().sum()


# In[100]:


# Applying One Hot Encoding to X_cat
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder()
X_cat=onehotencoder.fit_transform(X_cat).toarray()


# In[101]:


X_cat


# In[102]:


employee_df.info()


# In[103]:


#Group the other numerical values from training data into X_numerical
X_numerical=employee_df[['Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction', 'HourlyRate','JobInvolvement',   
'JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction', 'StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrProject Director']]


# In[104]:


X_numerical


# In[105]:


#convert previous X_cat from array to dataframe
X_cat=pd.DataFrame(X_cat)


# In[106]:


# Concantenate X_cat and X_numerical into X
X_all = pd.concat([X_cat,X_numerical],axis=1)


# In[107]:


X=X_all


# In[108]:


X


# In[109]:


# Looking at training data, the features seem imbalanced eg DailyRate vs Distance from Home
X.iloc[:,0:40]


# # MinMax scaling - to normalize variables to [0,1]

# In[110]:


#Need to scale otherwise DailyRate would take over lower values
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X=scaler.fit_transform(X_all)


# In[111]:


X


# In[112]:


y=employee_df['Attrition']


# In[113]:


y


# # Run Smote to handle imbalanced data

# In[114]:


# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


# In[115]:


# summarize distribution
counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()


# In[116]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)


# # Logistic Regression

# In[117]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()


# In[118]:


logreg.fit(X_train, y_train)


# In[119]:


y_pred = logreg.predict(X_test)


# In[120]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report


# In[121]:


cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[122]:



TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
print(np.asarray([['TN', 'FP'], ['FN', 'TP']]))


# In[123]:


sns.heatmap(cm,annot=True)


# In[124]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[125]:


print(classification_report(y_test, y_pred))


# In[126]:


# Generate the prediction values for each of the test observations using predict_proba() function rather than just predict
preds = logreg.predict_proba(X_test)[:,1]

# Store the false positive rate(fpr), true positive rate (tpr) in vectors for use in the graph
fpr, tpr, _ = metrics.roc_curve(y_test, preds)

# Store the Area Under the Curve (AUC) so we can annotate our graph with theis metric
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC Curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Log Regression')
plt.legend(loc = "lower right")
plt.show()


# # Support Vector Machine

# In[127]:


from sklearn.svm import SVC

svc_model = SVC(kernel='linear', probability=True)
svc_model.fit(X_train, y_train)
svc_model.score(X_test, y_test)

y_pred = svc_model.predict(X_test)


# In[128]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[129]:


TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
print(np.asarray([['TN', 'FP'], ['FN', 'TP']]))


# In[130]:


cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm,annot=True,fmt="d");

# Unable to predict 42 people who left, result is similar to logistic regression model.
#Predicted 52 people will leave, but all stayed.


# In[131]:


print(classification_report(y_test, y_pred))


# In[132]:


# Generate the prediction values for each of the test observations using predict_proba() function rather than just predict
preds = svc_model.predict_proba(X_test)[:,1]

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
plt.title('Receiver operating characteristic SVM')
plt.legend(loc = "lower right")
plt.show()


# # Naive Bayes

# In[133]:


#Create a Gaussian Classifier
nb = GaussianNB()

# Train the model using the training sets 
nb.fit(X_train, y_train)

#Predict Score 
y_pred = nb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True,fmt="d");
print(classification_report(y_test, y_pred))


# In[134]:


TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
print(np.asarray([['TN', 'FP'], ['FN', 'TP']]))


# In[135]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[136]:


# Generate the prediction values for each of the test observations using predict_proba() function rather than just predict
preds = nb.predict_proba(X_test)[:,1]

# Store the false positive rate(fpr), true positive rate (tpr) in vectors for use in the graph
fpr, tpr, _ = metrics.roc_curve(y_test, preds)

# Store the Area Under the Curve (AUC) so we can annotate our graph with theis metric
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC Curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='green', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Naive Bayes')
plt.legend(loc = "lower right")
plt.show()


# # Conclusion
# Choose between logistic and SVM because both have similar scores for accuracy, precision and recall.  In general, either model is 80% accurate in predicting attrition rate. While naive bayes is not far behind at 75%, it has a worse True Positive rate.  It misclassifies a larger group of people as leavers when they all want to stay.

# # GridSearch

# We initially ran into convergence problem for logistic regression and used grid search to identify max_iter rate.

# In[137]:


lr_params = {
    'penalty': ['l1','l2'],
    'C': [1, 10, 100]
}

lr_gs = GridSearchCV(LogisticRegression(max_iter=500, solver='liblinear'), lr_params, cv=5, verbose=1)
lr_gs.fit(X, y)


# In[138]:


lr_gs.best_score_


# In[139]:


lr_gs.best_params_


# In[140]:


# gridsearch SVM

svc_params = {
    'C': [1, 10, 100],
    'gamma': [0.001, 0.0001],
    'kernel': ['linear','rbf']
}

svc_gs = GridSearchCV(SVC(probability=True), svc_params, cv=5, verbose=1)
svc_gs.fit(X, y)


# In[141]:


svc_gs.best_score_


# In[142]:


svc_gs.best_params_


# In[143]:


from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



title = "Learning Curves (Logistic)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)

estimator = LogisticRegression(C = 70.17038286703837, penalty = 'l1', solver = 'liblinear')
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

title = r"Learning Curves (SVM, Linear kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)

estimator = SVC(C=27.825594022071257, gamma=1e-05, kernel = 'linear')
plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show();

