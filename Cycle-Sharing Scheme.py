#!/usr/bin/env python
# coding: utf-8

# In[1]:


# packages

get_ipython().run_line_magic('matplotlib', 'inline')

import random
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
import statistics
import numpy as np
import scipy
from scipy import stats
import seaborn
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# importing dataset(Source : Kaggle (https://www.kaggle.com/pronto/cycle-share-dataset/))
data = pd.read_csv('trip.csv' , error_bad_lines=False) # cause offending lines to be skipped 


# # Exploratory Data Analysis

# In[3]:


print(len(data))
data.head()


# ##  Univariate Analysis

# In[4]:


# determine time range of dataset
data = data.sort_values(by = 'starttime')
data.reset_index()
print('Data range of dataset: %s - %s'%(data.loc[1,'starttime'],data.loc[len(data)-1,'stoptime']))

The above analysis has given us insights that dataset we are dealing with is from October,2014 till September,2016.Also the analysis suggests that the cycle sharing service is beyond the regular 9 to 5 business hours. 
# ### Performing various Distribution studies in order to highlight the target audience

# In[5]:


# distribution on the basis of usertype
groupby_user = data.groupby('usertype').size() # size of each group
groupby_user.plot.bar(title = 'Distribution of user types')


# In[6]:


# distribution on the basis of gender
group_by_gender = data.groupby('gender').size()
group_by_gender.plot.bar(title = 'Distribution by Gender')


# In[7]:


# distribution by birth year
data = data.sort_values(by = 'birthyear')
groupby_birthyear = data.groupby('birthyear').size()
groupby_birthyear.plot.bar(title = 'Distribution by birth year',color = 'red',figsize = (15,4))


# In[8]:


# distribution by usertypes for specific birth year
data_mil = data[(data['birthyear'] >= 1977) & (data['birthyear'] <= 1994)]
groupby_mil = data_mil.groupby('usertype').size()
groupby_mil.plot.bar(title = 'Distribution by Usertype(with respect to birth year)',color = 'purple')


# ## Multivariate Analysis

# In[9]:


# plotting distribution of birth year by gender types
groupby_gender_birthyear = data.groupby(['birthyear','gender'])['birthyear'].count().unstack('gender').fillna(0)
groupby_gender_birthyear[['Male','Female','Other']].plot.bar(title = 'Distribution of Birth Year by Gender',stacked = True,figsize=(15,4))


# In[10]:


# distribution of birth year by usertype
groupby_birthyear_user = data.groupby(['birthyear','usertype'])['birthyear'].count().unstack('usertype').fillna(0)
groupby_birthyear_user['Member'].plot.bar(title = 'Distribution of Birth year by User type',stacked = True,figsize = (15,4),color='green')


# In[11]:


# validation if we don't have birth year available for Short term pass holders
# data[data['usertype'] == 'Short-Term Pass Holder']['birthyear'].isnull().values.all()


# In[12]:


# converting string to datetime and deriving new features
list_ = list(data['starttime'])
list_ = [datetime.datetime.strptime(x,"%m/%d/%Y %H:%M")for x in list_]
data['starttime_mod'] = pd.Series(list_,index = data.index)
data['starttime_date'] = pd.Series([x.date() for x in list_],index = data.index)
data['starttime_year'] = pd.Series([x.year for x in list_],index = data.index)
data['starttime_month'] = pd.Series([x.month for x in list_],index = data.index)
data['starttime_day'] = pd.Series([x.day for x in list_],index = data.index)
data['starttime_hour'] = pd.Series([x.hour for x in list_],index = data.index)


# ## Time Series Analysis

# In[13]:


data.groupby('starttime_date')['tripduration'].mean().plot.bar(title = 'Distribution of Trip Duration by date',figsize = (15,4)) 


# In[14]:


# Seasonality Distribution 


# ### Determining the measures of center using statistical packages
# #### Performed in order to determine the frequently visited stations that thereby can be utilized effectively for promotional campaigns.

# In[15]:


trip_duration = list(data['tripduration'])
station_from = list(data['from_station_name'])
# mean
print("Mean of trip duration %f"%statistics.mean(trip_duration))
# median
print("Median of trip duration %f"%statistics.median(trip_duration))
# mode : most trips originated from
print("Most trips originated from %s"%statistics.mode(station_from))

Why mean is almost double the median(might be all larger values are after the median ) ? lets look for the normal distribution plot to answer our query.
# In[16]:


data['tripduration'].plot.hist(bins=100,title = 'Frequency distribution of Trip duration' , color = 'purple')
plt.show()


# In[17]:


# the extremities are pulling the mean towards higher values keeping median with a pretty low value.


# In[18]:


# To look for outliers
# Plotting box plot for the same.
box = data.boxplot(column = ['tripduration']) # column is an attribute
plt.show()


# In[19]:


# determining ratio of values in observations of trip duration which are outliers 
q75 , q25 = np.percentile(trip_duration,[75,25])
iqr = q75-q25
print("Proportion of values as outliers : %f percent" %((len(data) - len([x for x in trip_duration if q75+(1.5*iqr)>=x>= q25-(1.5*iqr)]))*100/float(len(data))))

NOTE 1:
    Number of Outlier Values = Length of all values - Length of all non outlier values
    Ratio of outliers = Length of all outlier values/Length of all values * 100
# In[20]:


# as outlier value is very high that means the only option left is transformation
# calculating z scores for observations lying within trip_duration
mean_trip_duration = np.mean([x for x in trip_duration if q75+(1.5*iqr)>=x>= q25-(1.5*iqr)])
upper_whisker = q75+(1.5*iqr)
print("Mean of trip duration is %f"%mean_trip_duration)


# In[21]:


# transforming Dataset
def transform_trip_duration(x):
    if x > upper_whisker:
        return mean_trip_duration
    return x
data['tripduration_mean'] = data['tripduration'].apply(lambda x : transform_trip_duration(x))
data['tripduration_mean'].plot.hist(bins=100,title = 'Frequency distribution of mean transformed Trip Duration')
plt.show()


# In[22]:


# determining measures of center without outliers
print("Mean of trip duration is %f"%data['tripduration_mean'].mean())
print("Standard Deviation of trip duration is %f"%data['tripduration_mean'].std())
print("Median of trip duration is %f"%data['tripduration_mean'].median())

NOTE 2 :
    CORRELATIONS : 
    # PEARSON R CORRELATION (linearly related 2 variables)
    # KENDALL RANK CORRELATION (to determine the strength and direction of relationship between two quantitative features)
    # SPEARMAN RANK CORRELATION (t Spearman rank correlation does not make any assumptions about the distribution of the data. Spearman rank correlation is most suitable for ordinal data.)
# In[23]:


# pairplot AGE VS TRIPDURATION
data['age'] = data['starttime_year'] - data['birthyear']
data = data.dropna()
seaborn.pairplot(data,vars =['age','tripduration'],kind = 'reg')
plt.show()


# In[24]:


# calculating correlation
pd.set_option('display.width',100)
pd.set_option('precision',3)
correlations = data[['tripduration','age']].corr(method='pearson')
print(correlations)


# In[25]:


# computing two tail t-test of categories of gender 
for cat in ['gender']:
    
    print('Category: %s\n'%cat)
    groupby_category = data.groupby(['starttime_date', cat])['starttime_date'].count().unstack(cat)
    groupby_category = groupby_category.dropna()
    category_names = list(groupby_category.columns)
    print('Sub-category names : %s\n'%category_names)
                                                             
    for comb in [(category_names[i],category_names[j]) for i in range(len(category_names)) for j in range(i+1, len(category_names))]:
        print('%s %s'%(comb[0], comb[1]))
        t_statistics = stats.ttest_ind(list(groupby_category[comb[0]]),list(groupby_category[comb[1]]))
        print('Statistic :%f,P Value:%f'%(t_statistics.statistic, t_statistics.pvalue))
        print('\n')


# In[26]:


# script to validate the central limit theorem on trips dataset
daily_tickets = list(data.groupby('starttime_date').size())
sample_tickets = []
checkpoints = [1,10,100,300,500,1000]
plot_count = 1

random.shuffle(daily_tickets)

plt.figure(figsize=(15,7))
binrange=np.array(np.linspace(0,700,101))

for i in range(1000):
    if daily_tickets:
        sample_tickets.append(daily_tickets.pop())
        
    if i+1 in checkpoints or not daily_tickets:
        plt.subplot(2,3,plot_count)
        plt.hist(sample_tickets, binrange)
        plt.title('n=%d' % (i+1),fontsize=15) 
        plot_count+=1 
        
    if not daily_tickets:
        break
        
plt.show()

Hence, Central Limit theorem proved as with increasing sample size the distribution is moving towards normal distribution.
# ##### **CHALLENGE : Since the Short time pass customers dataset is very less and incomplete thereby more market research needs to be done in order to get significant insights regarding them.
# 
# ### CASE FINDINGS:
# ####  -> Pier 69/Alaskan Way & Clay St. were the best station to kick off campaign promotions.
# ####  -> Most of the cusomers are males of a specific age group(1994-1997 born) are our target audience thereby promotional campaigns must be largely targeted for this group but yes simulatneously we must look for approaches or offers for other groups also just to increase their portion in ou customer count.
# ####  -> Outliers are tiny so we cannot remove them directly so these must be transformed.
# ####  -> Age and tripduration are also positively related (95% confidence level).
# ####  -> Central limit theorem was proved in order to define the significance of large dataset(atleast optimal such as not in the case of Short time pass customers.

# In[ ]:




