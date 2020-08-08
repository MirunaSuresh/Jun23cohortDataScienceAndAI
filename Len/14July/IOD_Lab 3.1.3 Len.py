#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src=https://www.institutedata.com/wp-content/uploads/2019/10/iod_h_tp_primary_c.svg width="300">
# </div>

# # Lab 3.1.3
# 
# ## Data
# 
# The Philippine Statistics Authority (PSA) spearheads the conduct of the Family Income and Expenditure Survey (FIES) nationwide. The survey, which is undertaken every three (3) years, is aimed at providing data on family income and expenditure, including, among others, levels of consumption by item of expenditure, sources of income in cash, and related information affecting income and expenditure levels and patterns in the Philippines.
# 
# You can download the data from [here](https://www.kaggle.com/grosvenpaul/family-income-and-expenditure).
# 
# The purpose of today's lab is to use simulation to visualize the sampling distribution for the sample mean. The Central Limit Theorem (CLT) tells us that as our sample size gets larger, the sampling distribution of the sample mean converges to a normal distribution. Therefore, when we have a large sample size, we can say that the sampling distribution for the sample mean is approximately normal, regardless of the distribution from which we are sampling.
# 
# Let's start by taking a look at the data, **`Total Household Income`** will serve as a "population" for the purposes of this lab. 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


house_hold = pd.read_csv('/Users/lenkwok/Desktop/projects/Family_Income_and_Expenditure.csv')
# Read CSV


# In[6]:


house_hold.head()


# In[7]:


house_hold.tail()


# In[9]:


house_hold.shape


# In[10]:


house_hold.dtypes


# #### Surveying the populations
# 
# ##### 1. Create a histogram of `Total Household Income`.

# In[5]:


sns.distplot(house_hold['Total Household Income'])


# ##### 2. How would you describe the shape of this population?

# **ANSWER:**skewed to the left

# ##### 2. What is the mean income of this population?

# In[14]:


mean_income=house_hold['Total Household Income'].mean()
print('Mean income is', mean_income)


# ##### 3. What sampling statistic/point estimate would you use to estimate the mean of this population if you were given a random sample from the population?

# **ANSWER:**

# #### Simulated sampling (sample means)

# Now, we'd like to get an idea of what happens when we take multiple random samples of size 5. 
# 
# Take 10 sample (size=5) from the entire population. Calculate means for each sample. Now make a histogram of all the sample means.
# 
# - Describe the shape of the histogram.
# - What is the center of the distribution of sample means?

# ##### 4. Simulation with `sample_size=5`

# Take 10 samples, but with a sample size of 5.

# ###### 4.A Make a histogram of all the sample means

# In[ ]:


# ANSWER


# ###### 4.B Describe the shape of the histogram.

# **ANSWER:**

# ##### 7. Simulation with `sample_size=15`

# let's try taking another 1000 samples, but with a sample size of 15

# In[ ]:


# ANSWER


# ##### 8. Simulation with `sample_size=50`

# Let's try taking another 1000 samples, but with a sample size of 50

# In[ ]:


# ANSWER


# ###### 8.A Describe the shape of the histogram of sample means (using sample size of 50)

# **ANSWER:**

# ###### 8.B What is mean of the distribution of sample means?

# In[ ]:


# ANSWER


# **ANSWER:**
# 
# > If repeated random samples of a given size n are taken from a population of values for a quantitative variable, where the population mean is μ (mu) and the population standard deviation is σ (sigma) then the mean of all sample means (x-bars) is population mean μ (mu).

# ###### 8.C As the sample size grew, did your results confirm the CLT?

# **ANSWER:**

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
