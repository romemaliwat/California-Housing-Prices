#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('housing.csv')
df


# In[3]:


df.info()


# In[4]:


df.dropna(inplace = True) #Remove all missing values since 207 is not that significant compared to 20640 instances.
df


# In[5]:


df.info()


# In[6]:


from sklearn.model_selection import train_test_split

X = df.drop(['median_house_value'], axis = 1) # Independent variables should include necessary columns except for the target variable (median_house_value)
y = df['median_house_value'] # Dependent variable should include only the target variable (median_house_value)


# In[7]:


X


# In[8]:


y


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
# SPlit the data into training dataset (X_train, and y_train), and testing dataset (X_test, and y_test)
# test_size=0.2 mean that the 20% (randomized) of the whole dataset will be used for testing data.


# In[10]:


train_data = X_train.join(y_train)
train_data


# In[11]:


# 16346 rows are the 80% of the whole dataset, they are now used for training data
train_data.hist(figsize=(15,8)) # Use histogram to see the distribution of the data in each column


# In[12]:


train_data.corr(numeric_only=True) # Use correlation to see the numerical value of the relationship of every two variables in the data


# In[13]:


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap='YlGnBu')
# Use heatmap to see the correlation strength of two variables with colors
# annot = True would include the numerical value of the correlation


# In[14]:


train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)

# Since the recorded data for these columns are by block, some have huge values while others are small.
# To scale them down, and make the distribution (refer to histogram) make more look 'normal', we scale them down using np.log (logarithm).

train_data.hist(figsize=(15,8))


# In[15]:


train_data.ocean_proximity.value_counts()
# We'll see the categories of the ocean_proximity column and their numbers recorded (rows).
# Since it is a categorical, the best way to use is to encode it (get_dummies).


# In[16]:


pd.get_dummies(train_data['ocean_proximity'], dtype = int)
# We'll get the categories of ocean_proximity and make them individual columns
# By default, it would return a True or False value, hence we use the dtype=int to make them binary (0 and 1).
# In linear regression, everything has to be numerical that's why we use 0 and 1 value.


# In[17]:


# Now, let's join the original train_data with the ocean_proximity categories encoded as column, and the ocean_proximity column removed.
train_data = train_data.join(pd.get_dummies(train_data['ocean_proximity'], dtype=int)).drop(['ocean_proximity'], axis=1)


# In[18]:


train_data


# In[19]:


# Now, let visualize and see the correlation of every two columns of the train_data
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap='YlGnBu')


# In[20]:


# By using the scatterplot, we can see if there is a relationship between median price value and the proximity of the houses in the ocean
plt.figure(figsize=(15,8))
sns.scatterplot(x='latitude', y='longitude', data=train_data, hue='median_house_value', palette='coolwarm')


# In[21]:


# As we can see in the scatterplot, in general, the nearer the houses to the ocean, the higher their values.
# Now, we can add another variables (bedroom ratio and household rooms) just to see if these have also significant effect to the house prices.
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']


# In[22]:


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap='YlGnBu')


# In[23]:


# Now, we can proceed to linear regression!
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler #for scaling, in case scaling the values has an impact to the accuracy

scaler = StandardScaler() #typical, start with instance of the function

X_train, y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value'] #split the training data (train_data) into independent valiables and target variable (median_house_value)
X_train_s = scaler.fit_transform(X_train) # store the scaled train data to X_train_s


# In[24]:


# Remember that the testing data hasn't been encoded yet (the ocean_proximity) column is still there.
# To match it with the training data, it must be encoded first, then scaled.

# Now, let's join the original train_data with the ocean_proximity categories encoded as column, and the ocean_proximity column removed.
# Basically, just do whatever we did to the train data

test_data = X_test.join(y_test)

test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)

test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']

test_data = test_data.join(pd.get_dummies(test_data['ocean_proximity'], dtype=int)).drop(['ocean_proximity'], axis=1)


# In[29]:


# Now, we can split the independent valriables and dependent variable of the test data so we scale them

X_test, y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']
# Since the 'ISLAND' column is missing in the test data, we have to use X_train.align to fill the 'ISLAND' column back.
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0) 
# This ensures that the column in train and test have exactly the same column.
# It fills out the 'ISLAND' value with 0 instead.
X_test_s = scaler.transform(X_test)


# In[32]:


reg = LinearRegression()
reg.fit(X_test, y_test)
reg.score(X_test, y_test)


# In[33]:


reg.fit(X_test_s, y_test)
reg.score(X_test_s, y_test)


# In[38]:


y_pred_0 = reg.predict(X_test_s[:1])  # first row
print("Predicted median_house_value:", y_pred_0[0])
print("Actual median_house_value:", y_test.values[0])


# In[40]:


y_pred_1 = reg.predict(X_test_s[:2])  # second row
print("Predicted median_house_value:", y_pred_1[1])
print("Actual median_house_value:", y_test.values[1])


# In[41]:


y_pred_2 = reg.predict(X_test_s[:3])  # third row
print("Predicted median_house_value:", y_pred_2[2])
print("Actual median_house_value:", y_test.values[2])


# In[ ]:




