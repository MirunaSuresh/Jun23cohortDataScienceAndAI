#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import data from Google Drive shared folder into jupyter notebook
 get data summary
 find average or min of one of the numeric columns
 write a function to print the shape
write a function to calculate average of numeric field


# In[3]:


import pandas as pd
df = pd.read_csv ('/Users/lenkwok/Downloads/AirPassengers.csv')
print (df)


# In[5]:


print(len(df))


# In[6]:


print(len(df.columns))


# In[12]:


Total = df['Passengers'].sum()


# In[16]:


Average=Total/len(df)
print(Average)


# In[8]:


print(df.shape)


# In[7]:


df.loc[:,"Passengers"].mean()

