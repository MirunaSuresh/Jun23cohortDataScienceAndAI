#!/usr/bin/env python
# coding: utf-8

# https://towardsdatascience.com/logistic-regression-a-simplified-approach-using-python-c4bc81a87c31

# In[122]:


import numpy as np
import pandas as pd


# In[123]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[124]:


#For Dummy variables
train=pd.read_csv('/Users/lenkwok/Downloads/projects/titanic.csv')
train.head()


# In[125]:


train.isnull()


# In[126]:


sns.heatmap(train.isnull())


# In[39]:


#count plot that shows the number of people who survived which is our target variable
sns.countplot(x='Survived',data=train)


# In[8]:


#Further, we can plot count plots on the basis of gender and passenger class.
sns.countplot(x='Survived',hue='Sex',data=train)


# In[10]:


#we can infer that passengers belonging to class 3 died the most.
sns.countplot(x='Survived',hue='Pclass',data=train)


# Data Cleaning
# 

# In[11]:


#check the average age by passenger class.
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[58]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[127]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[205]:


train['Age']


# # practice using Simple Imputer from sklearn

# In[209]:


#Create copy of train DF just in case
df=train.copy()
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')
imputer.fit_transform(df[['Age']])


# In[70]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# In[54]:


train.drop('Cabin',axis=1,inplace=True)


# # Converting Categorical Features
# 

# In[128]:


train.info()


# In[129]:


#dummying the sex and embark columns, drop first columnn for both
#Note sex and embark are different from Sex and Embarked
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[130]:


sex


# In[131]:


embark


# In[132]:


#After dummying, we will drop the rest of the columns which are not needed.
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[42]:


train


# In[61]:


train = pd.concat([train,sex,embark],axis=1)


# In[62]:


train


# # Test Train Split
# 

# In[63]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# # Training and Predicting
# 

# In[67]:


from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


# In[66]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# # For Label encoding

# In[191]:


new_df


# In[192]:


#Label encoding for sex only
from sklearn import preprocessing
label_encoder_sex = preprocessing.LabelEncoder()
label_encoder_sex.fit_transform(new_df['Sex'])


# In[190]:


#Label encoding for sex only
from sklearn import preprocessing
label_encoder_embarked = preprocessing.LabelEncoder()
label_encoder_embarked.fit_transform(new_df['Embarked'])


# # For One Hot Encoding

# In[193]:


from sklearn.preprocessing import OneHotEncoder
onehot_encoder_sex=OneHotEncoder(sparse=False)
 
onehot_encoder_sex=onehot_encoder.fit_transform(new_df[['Sex']])


# In[168]:


onehot_encoder_sex


# In[194]:


new_df


# In[195]:


onehot_encoder_embarked = new_df.dropna(subset=['Embarked'])


# In[196]:


from sklearn.preprocessing import OneHotEncoder
onehot_encoder_embarked=OneHotEncoder(sparse=False)
 
onehot_encoder_embarked.fit_transform(new_df[['Embarked']])


# In[135]:


pip install category_encoders


# # target encoding

# In[142]:


from category_encoders import TargetEncoder


# In[189]:


new_df


# In[197]:


encoder = TargetEncoder()
encoder.fit_transform(new_df['Sex'],new_df['Survived'])


# In[198]:


encoder = TargetEncoder()
encoder.fit_transform(new_df['Embarked'],new_df['Survived'])


# In[ ]:




