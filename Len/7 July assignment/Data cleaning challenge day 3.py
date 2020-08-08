#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import datetime


# In[21]:


earthquakes = pd.read_csv("/Users/lenkwok/Desktop/projects/earthquakes.csv")


# In[4]:


earthquakes.head()


# In[22]:


earthquakes['Date'].dtype


# In[23]:


# Convert date String to datetime format and add to column date_parsed
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y", infer_datetime_format= True, utc = True)


# In[24]:


earthquakes['date_parsed'].head()


# In[26]:


# Check column date_pased
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day


# In[27]:


day_of_month_earthquakes


# In[28]:


# Plot day of month
day_of_month_earthquakes = day_of_month_earthquakes.dropna()


# In[29]:


sns.distplot(day_of_month_earthquakes, kde=False, bins=31)


# In[ ]:




