#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
from IPython.display import HTML
import warnings
pd.set_option('max_columns', 100)
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
my_pal = sns.color_palette(n_colors=10)


# In[3]:


train = pd.read_csv('/Users/lenkwok/Desktop/2019 bowl/data-science-bowl-2019/train.csv')
train_labels = pd.read_csv('/Users/lenkwok/Desktop/2019 bowl/data-science-bowl-2019/train_labels.csv')
test = pd.read_csv('/Users/lenkwok/Desktop/2019 bowl/data-science-bowl-2019/test.csv')
specs = pd.read_csv('/Users/lenkwok/Desktop/2019 bowl/data-science-bowl-2019/specs.csv')
ss = pd.read_csv('/Users/lenkwok/Desktop/2019 bowl/data-science-bowl-2019/sample_submission.csv')


# In[4]:


train_ = train.sample(1000000) #sample 1M observations


# #The outcomes in this competition are grouped into 4 groups (labeled accuracy_group in the data):
# 3: the assessment was solved on the first attempt
# 2: the assessment was solved on the second attempt
# 1: the assessment was solved after 3 or more attempts
# 0: the assessment was never solved

# In[5]:


train_labels.head()


# In[24]:



train_labels.groupby('accuracy_group')['game_session'].count().plot(kind='barh', figsize=(15, 5), title='Target (accuracy_group)')
plt.show()


# #Things to note about the taget:
# 
# Accuracy of 100% goes to group 3
# Accuracy of ~50% goes to group 2
# Not finishing goes to group 0
# Group 1 looks http://localhost:8888/notebooks/2019%20Data%20science%20bowl%20Len.ipynb#to have the most variation

# In[10]:


sns.pairplot(train_labels, hue='accuracy_group')
plt.show()


# # train.csv / test.csv
# The data provided in these files are as follows:
# - `event_id` - Randomly generated unique identifier for the event type. Maps to event_id column in specs table.
# - `game_session` - Randomly generated unique identifier grouping events within a single game or video play session.
# - `timestamp` - Client-generated datetime
# - `event_data` - Semi-structured JSON formatted string containing the events parameters. Default fields are: event_count, event_code, and game_time; otherwise - fields are determined by the event type.
# - `installation_id` - Randomly generated unique identifier grouping game sessions within a single installed application instance.
# - `event_count` - Incremental counter of events within a game session (offset at 1). Extracted from event_data.
# - `event_code` - Identifier of the event 'class'. Unique per game, but may be duplicated across games. E.g. event code '2000' always identifies the 'Start Game' event for all games. Extracted from event_data.
# - `game_time` - Time in milliseconds since the start of the game session. Extracted from event_data.
# - `title` - Title of the game or video.
# - `type` - Media type of the game or video. Possible values are: 'Game', 'Assessment', 'Activity', 'Clip'.
# - `world` - The section of the application the game or video belongs to. Helpful to identify the educational curriculum goals of the media. Possible values are: 'NONE' (at the app's start screen), TREETOPCITY' (Length/Height), 'MAGMAPEAK' (Capacity/Displacement), 'CRYSTALCAVES' (Weight).

# In[11]:


train.head()


# #event_id & game_session.  They say it's randomly generated, but is that true? Looks to be hex, /
# lets convert it to an integer. Plotting shows nothign really interesting.

# In[12]:


train['event_id_as_int'] = train['event_id'].apply(lambda x: int(x, 16))
train['game_session_as_int'] = train['game_session'].apply(lambda x: int(x, 16))


# ## timestamp
# Lets see how many observations we have over time. Are they all in the same/similar time zone?
# - Looks like number of observations rises over time. Steep pickup and dropoff at the start/end
# - Much less use during the middle of the night hours. Use increases during the day with a slow reduction in use around midnight. We don't know how the timestamp relates to time zones for different users.
# - More users on Thursday and Friday. 

# In[14]:


# Format and make date / hour features
train['timestamp'] = pd.to_datetime(train['timestamp'])
train['date'] = train['timestamp'].dt.date
train['hour'] = train['timestamp'].dt.hour
train['weekday_name'] = train['timestamp'].dt.weekday_name
# Same for test
test['timestamp'] = pd.to_datetime(test['timestamp'])
test['date'] = test['timestamp'].dt.date
test['hour'] = test['timestamp'].dt.hour
test['weekday_name'] = test['timestamp'].dt.weekday_name


# In[15]:


print(f'Train data has shape: {train.shape}')
print(f'Test data has shape: {test.shape}')


# In[ ]:


train.groupby('date')['event_id']     .agg('count')     .plot(figsize=(15, 3),
         title='Numer of Event Observations by Date',
         color=my_pal[2])
plt.show()
train.groupby('hour')['event_id']     .agg('count')     .plot(figsize=(15, 3),
         title='Numer of Event Observations by Hour',
         color=my_pal[1])
plt.show()
train.groupby('weekday_name')['event_id']     .agg('count').T[['Monday','Tuesday','Wednesday',
                     'Thursday','Friday','Saturday',
                     'Sunday']].T.plot(figsize=(15, 3),
                                       title='Numer of Event Observations by Day of Week',
                                       color=my_pal[3])
plt.show()


# # event_data
# This looks to have most of the interesting data about the event. It is in JSON format which isn't easy to wrangle in a tabular way. We need to be clever when parsing this data. They have already parsed some of this data for us like `event_count` and `event_code`.

# In[19]:


print(train['event_data'][4])
print(train['event_data'][5])


# ## installation_id *important - predictions are grouped by these*
# - Randomly generated unique identifier grouping game sessions within a single installed application instance.
# - We will be predicting based off of these IDs
# - The training set has exactly 17000 unique `installation_ids`

# In[20]:


train['installation_id'].nunique()


# In[23]:


train.groupby('installation_id').count()['event_id'].plot(kind='hist',bins=40,color=my_pal[4],figsize=(15, 5),title='Count of Observations by installation_id')
plt.show()


# Lets take a log transform of this count to we can more easily see what the distribution of counts by `insallation_id` looks like

# In[25]:


train.groupby('installation_id').count()['event_id'].apply(np.log1p).plot(kind='hist',bins=40,color=my_pal[6],figsize=(15, 5),title='Log(Count) of Observations by installation_id')
plt.show()


# Lets looks at some of the installation_ids with the highest counts. We see some installation_ids have tens of thousands of observations!

# train.groupby('installation_id').count()['event_id'].sort_values(ascending=False).head(5)

# # # Wow, 50000+ events for a single `installation_id`. Lets take a closer look at the id with the most observations. Not exactly sure what I'm looking at here. But it looks like this `installation_id` spans a long duration (over one month). Could this be installed by a bot? The use history does not look natural.

# In[28]:


train.query('installation_id == "f1c21eda"').set_index('timestamp')['event_code']     .plot(figsize=(15, 5), title='installation_id #f1c21eda event Id - event code vs time',style='.',color=my_pal[8])
plt.show()


# ## event_code
# - Identifier of the event 'class'. Unique per game, but may be duplicated across games. E.g. event code '2000' always identifies the 'Start Game' event for all games. Extracted from event_data.

# In[33]:


train.groupby('event_code')     .count()['event_id']     .sort_values()     .plot(kind='bar',
         figsize=(15, 5),
         title='Count of different event codes.')
plt.show()


# lets take a closer look at the event codes `4070` and `4030`
# - We notice that event 4070 and 4030 always comes with coordinates (x, y) and stage_width.
# - Possibly they could be marking acheivements or something related to position on the screen.
# These events look like this:
# ```
# {"size":0,"coordinates":{"x":782,"y":207,"stage_width":1015,"stage_height":762},"event_count":55,"game_time":34324,"event_code":4030}
# ```

# ## game_time
# - Time in milliseconds since the start of the game session. Extracted from event_data.
# - The `log1p` transform shows a somewhat normal distribution with a peak at zero.

# In[35]:


train['game_time'].apply(np.log1p).plot(kind='hist',figsize=(15, 5),bins=100, title='Log Transform of game_time',
color=my_pal[1])
plt.show()


# In[ ]:




