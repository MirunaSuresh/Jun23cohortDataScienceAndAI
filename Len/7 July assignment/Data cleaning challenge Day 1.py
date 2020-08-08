#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


sf_permits = pd.read_csv("/Users/lenkwok/Downloads/Building_Permits.csv")


# In[4]:


#check for 1st five rows, we can see some missing values, denoted by Na
sf_permits.head()


# In[5]:


missing_values_count = sf_permits.isnull().sum()


# In[6]:


# prints out total number of missing values in each column.
missing_values_count


# In[7]:


# how many total missing values do we have?
total_cells = np.product(sf_permits.shape)
total_missing = missing_values_count.sum()


# In[8]:


#check output of total cells
total_cells


# In[9]:


#check output of total missing
total_missing


# In[10]:


# percent of data that is missing
(total_missing/total_cells) * 100


# In[11]:


# get the number of missing data points per column
missing_values_count_sf = sf_permits.isnull().sum()


# In[13]:


missing_values_count_sf


# In[22]:


#removing all the rows from the sf_permits dataset that contain missing value
missing_rows_gone=sf_permits.dropna()


# In[23]:


#removing all the columns with empty values
missing_col_gone=sf_permits.dropna(axis=1)
                  


# In[37]:


#print how many rows and columns were dropped
print(missing_rows_gone.shape[1])
print(missing_col_gone.shape[1])


# In[38]:


#fill nan value to 0 
sf_permits.fillna(0)


# In[39]:


sf_permits.fillna(method='bfill', axis=0).fillna(0)


# In[ ]:




