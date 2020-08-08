#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src=https://www.institutedata.com/wp-content/uploads/2019/10/iod_h_tp_primary_c.svg width="300">
# </div>

# # Data
# 
# > The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# > One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this lab, we'll explore this dataset to find insight.
# 
# [Titanic Dataset](https://www.kaggle.com/c/titanic/data)

# # Data Dictionary
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

# # Loading Modules

# In[60]:


# Load necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading Dataset
# 
# Read titanic dataset.

# In[61]:


# Read Titanic Dataset
titanic_csv = '/Users/lenkwok/Downloads/titanic.csv'
titanic = pd.read_csv(titanic_csv)


# # Explore Dataset

# In[4]:


#check column names
titanic.columns


# ## Head

# In[5]:


# Check Head
titanic.head()


# ## Tail

# In[4]:


# Check Tail
titanic.tail()


# ## Shape
# 
# Find shape of dataset.

# In[5]:


# ANSWER
titanic.shape


# ## Check Types of Data
# 
# Check types of data you have

# In[67]:


# ANSWER
titanic.dtypes


# ## Check Null Values
# 
# Check whether dataset have any null values.

# In[7]:


# ANSWER
titanic.isnull().sum()


# ## Fill Null Values
# 
# Is there any null values in any columns? 
# 
# - Identify those columns
# - Fill those null values using your own logic
#     - State your logic behind every steps

# In[69]:


def bar_chart(feature):
survived = titanic[titanic['Survived']==1][feature].value_counts()
dead = titanic[titanic['Survived']==0][feature].value_counts()
df=pd.titanic([survived,dead])
df.index =['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=D(10,5))


# ### Age

# In[3]:


# Find value in Age column with NaN
titanic[titanic['Age'].isna()]


# So, There are 177 rows have missing `Age` values. We can use median values of `Male` & `Female` to fill those values.

# In[4]:


titanic['Age'].fillna(titanic.groupby(by=['Sex'])['Age'].transform("median"), inplace=True) 


# In[20]:


# Fill in missing age values with mean.  
#find the average age of the passengers
ave_age = titanic.Age.mean()
ave_age


# ### Cabin

# In[7]:


titanic[titanic['Cabin'].isna()]


# In[8]:


#Let's look at Cabin column in greater detail include number of counts
titanic['Cabin'].value_counts()


# In[ ]:


#Seems that cabin has variatio in letter followed by two or three numbers.  
#Perhaps we can remove numbers from cabin.


# In[9]:


# Consider only the  first character as cabin number
titanic['Cabin'] = titanic['Cabin'].apply(lambda x: x[:1] if type(x) is str else x)


# In[10]:


# Check Cabin
titanic['Cabin'].value_counts()


# In[11]:


#Plot histograms side by side
titanic.groupby(by=['Pclass', 'Cabin']).agg({'Cabin': 'count'}).unstack().plot(kind='bar', figsize=(10,8));


# In[ ]:


#From the plot cabins `A`, `B`, `C` & `T` only exist in Pclass `1`.


# In[ ]:


cabin_map = {
    'A': 1
    , 'B': 2
    , 'C': 3
    , 'D': 4
    , 'E': 5
    , 'F': 6
    , 'G': 7
    , 'T': 8
}
titanic['Cabin'] = titanic['Cabin'].map(cabin_map)


# In[ ]:


# Fill Cabin with Mean values
titanic['Cabin'].fillna(titanic.groupby(by=['Pclass'])['Cabin'].transform("mean"), inplace=True) 


# In[ ]:


# Remove Decimal Numbers
titanic['Cabin'] = np.round(titanic['Cabin'], decimals=0)


# In[ ]:


# Check Cabin
titanic['Cabin'].value_counts()


# ### Embarked

# In[13]:


titanic[titanic['Embarked'].isna()]


# In[14]:


titanic['Embarked'].value_counts(normalize=True)


# Since 72% Passenger embarked from `S`. We can fill 2 rows of null values with `S`.

# In[58]:


titanic['Embarked']=titanic['Embarked'].replace('NaN','S')


# In[59]:


titanic[titanic['Embarked'].isna()]


# In[15]:


titanic['Embarked'] = titanic['Embarked'].apply(lambda x: x if type(x) is str else 'S')


# # Describe
# 
# Describe your dataset.

# In[13]:


titanic.info()


# In[9]:


# ANSWER
titanic.describe()


# # Relationship between Features and Survival
# 
# Find relationship between categorical features and survived.
# 
# **Describe your findings.**
# 
# #Pclass
# #Sex
# #SibSp
# #Parch
# #Embarked
# #Cabin

# In[18]:


# this code couldn't work....
def bar_chart(feature):
    survived=titanic.(titanic['Survived']==1[feature].value_counts()
    dead=titanic.(titanic['Survived']==0[feature].value_counts()
    df=pd.titanic([survived,dead])
    df.index=['Surved','Dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))


# In[19]:


def bar_charts(df, feature):
    '''
    Inputs:
    df: Dataset
    feature: Name of Feature to Check With Survived
    '''
    _agg = {
        'PassengerId': 'count'
    }
    _groupby = ['Survived', feature]

    df_feature = df.groupby(by=_groupby).agg(_agg)
    
    ax = df_feature.unstack().plot(kind='bar', figsize=(15,6))
    plt.legend(list(df_feature.index.levels[1].unique()))
    plt.xlabel('Survived')
    plt.xticks(np.arange(2), ('No', 'Yes'))
    plt.show();


# ## Pclass
# 
# Use barchart to find relationship between survived and pclass.  Note your findings.

# In[20]:


bar_charts(titanic, 'Pclass')


# In[28]:


#This is my earlier plot, which looks way different.
df1.plot.bar();
plt.show()


# ## Sex
# 
# Use barchart to find relationship between survived and sex.  Note your findings.

# In[21]:


bar_charts(titanic, 'Sex')


# ## Parch
# 
# Parch = Number of parents of children travelling with each passenger.

# In[22]:


bar_charts(titanic, 'Parch')


# ## SibSp

# In[23]:


bar_charts(titanic, 'SibSp')


# ## Embarked

# In[24]:


bar_charts(titanic, 'Embarked')


# # Feature Engineering
# 
# Create some new features from existing feature.

# ## Fare Class
# 
# Create a new class based on their fare. Is there any relationship between fare and survival? 

# In[25]:


def create_fare_class(x):
    if x > 30:
        fare_class = 1
    elif x > 20 and x <= 30:
        fare_class = 2
    elif x > 10 and x <= 20:
        fare_class = 3
    else:
        fare_class = 4
    return fare_class


# In[28]:


# ANSWER
titanic['FareClass'] = titanic['Fare'].apply(create_fare_class)


# In[29]:


bar_charts(titanic, 'FareClass')


# ## Age Class

# In[30]:


titanic['Age'].value_counts()


# In[34]:


def create_age_class(x):
    if x > 60:
        age_class = 5
    elif x > 35 and x <= 60:
        age_class = 4
    elif x > 25 and x <= 35:
        age_class = 3
    elif x > 16 and x <= 25:
        age_class = 2
    else:
        age_class = 1
    return age_class


# In[35]:


# apply new column to a dataframe
titanic['AgeClass'] = titanic['Age'].apply(create_age_class)


# In[37]:


bar_charts(titanic, 'AgeClass')


# # Staistical Overview

# In[38]:


from scipy import stats


# ## Correlation
# 
# Find correlation between `survived` and other features.

# In[39]:


titanic.corr()


# # [BONUS] Hypothesis Testing
# ---
# Hypothesis testing is the use of statistics to determine the probability that a given hypothesis is true. The usual process of hypothesis testing consists of four steps.
# 
# 1. Formulate the null hypothesis H_0 (commonly, that the observations are the result of pure chance) and the alternative hypothesis H_a (commonly, that the observations show a real effect combined with a component of chance variation).
# 
# 2. Identify a test statistic that can be used to assess the truth of the null hypothesis.
# 
# 3. Compute the P-value, which is the probability that a test statistic at least as significant as the one observed would be obtained assuming that the null hypothesis were true. The smaller the P-value, the stronger the evidence against the null hypothesis.
# 
# 4. Compare the p-value to an acceptable significance value  alpha (sometimes called an alpha value). If p<=alpha, that the observed effect is statistically significant, the null hypothesis is ruled out, and the alternative hypothesis is valid.

# ### Define Hypothesis
# 
# > Formulate the null hypothesis H_0 (commonly, that the observations are the result of pure chance) and the alternative hypothesis H_a (commonly, that the observations show a real effect combined with a component of chance variation).
# 
#     Null Hypothesis (H0): There is no difference in the survival rate between the young and old passengers.
# 
#     Alternative Hypothesis (HA): There is a difference in the survival rate between the young and old passengers.

# ### Collect Data
# 
# Next step is to collect data for each population group. 
# 
# Collect two sets of data, one with the passenger greater than 35 years of age and another one with the passenger younger than 35. The sample size should ideally be the same but it can be different. Lets say that the sample sizes is 100.

# In[40]:


N = 100
age = 35


# In[41]:


titanic_young = titanic[titanic['Age'] <= age].sample(N, random_state=42)
titanic_old = titanic[titanic['Age'] > age].sample(N, random_state=42)


# In[42]:


titanic_young['Survived'].value_counts()


# In[43]:


titanic_old['Survived'].value_counts()


# ### Set alpha (Let alpha = 0.05)
# 
# For example, if you want to be 95 percent confident that your analysis is correct, the alpha level would be 1 – . 95 = 5 percent, assuming you had a one tailed test. For two-tailed tests, divide the alpha level by 2.

# In[44]:


titanic_old['Survived'].value_counts()


# ### Calculate point estimate

# In[54]:


alpha=0.05
a = titanic_young['Survived']
b = titanic_old['Survived']


# In[46]:


## Calculate the variance to get the standard deviation
var_a = a.var()
var_b = b.var()

## Calculate the Standard Deviation
s = np.sqrt((var_a + var_b)/2)


# ### Calculate test statistic

# In[47]:


## Calculate the t-statistics
t = (a.mean() - b.mean())/(s*np.sqrt(2/N))


# ### Find the p-value
# 
# > Compute the P-value, which is the probability that a test statistic at least as significant as the one observed would be obtained assuming that the null hypothesis were true. The smaller the P-value, the stronger the evidence against the null hypothesis.

# In[48]:


## Compare with the critical t-value
## Degrees of freedom
df = 2*N - 2

#p-value after comparison with the t 
p = 1 - stats.t.cdf(t,df=df)


# In[49]:


print("t = " + str(t))
print("p = " + str(2*p))


# ### Interpret results
# 
# > Compare the p-value to an acceptable significance value  alpha (sometimes called an alpha value). If p<=alpha, that the observed effect is statistically significant, the null hypothesis is ruled out, and the alternative hypothesis is valid.

# In[56]:


def print_sig(p_value, alpha):
    if p_value < alpha:
        print("We reject our null hypothesis.")
    elif p_value > alpha:
        print("We fail to reject our null hypothesis.")
    else:
        print("Our test is inconclusive.")


# In[55]:


print_sig(p, alpha)


# In[57]:


## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(a,b)
print("t = " + str(t2))
print("p = " + str(p2))

print_sig(p2, alpha)


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
