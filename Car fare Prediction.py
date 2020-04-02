#!/usr/bin/env python
# coding: utf-8

# In[205]:


#Importing required libraries
import os 
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[2]:


os.chdir("D:\Python\Car Prediction")
os.getcwd()


# In[3]:


train= pd.read_csv("train_cab.csv")
train.head(5)


# In[5]:


test=pd.read_csv("test.csv")
test.head(5)


# In[6]:


#checking the number of rows and columns in train and test data
train.shape


# In[7]:


test.shape


# In[8]:


train.dtypes #checking the data-types in training dataset


# In[9]:


test.dtypes


# In[10]:


train.describe()


# In[11]:


test.describe()


# In[12]:


#Data Cleaning & Missing Value Analysis :
#Convert fare_amount from object to numeric
train["fare_amount"] = pd.to_numeric(train["fare_amount"],errors = "coerce")    #Using errors=’coerce’. It will replace all non-numeric values with NaN.  


# In[13]:


train.dtypes


# In[14]:


train.shape


# In[15]:


train.dropna(subset= ["pickup_datetime"]) 


# In[22]:


# Here pickup_datetime variable is in object so we need to change its data type to datetime
train.pickup_datetime =  pd.to_datetime(train.pickup_datetime,errors='coerce')


# In[24]:


### we will saperate the Pickup_datetime column into separate field like year, month, day of the week, etc

train['year'] = train['pickup_datetime'].dt.year
train['Month'] = train['pickup_datetime'].dt.month
train['Date'] = train['pickup_datetime'].dt.day
train['Day'] = train['pickup_datetime'].dt.dayofweek
train['Hour'] = train['pickup_datetime'].dt.hour
train['Minute'] = train['pickup_datetime'].dt.minute
train['Second'] = train['pickup_datetime'].dt.second


# In[26]:


train.dtypes


# In[27]:


#removing datetime missing values rows
train = train.drop(train[train['pickup_datetime'].isnull()].index, axis=0)
train.shape


# In[28]:


train["passenger_count"].describe()


# In[29]:


train = train.drop(train[train["passenger_count"]> 6 ].index, axis=0)


# In[30]:


#Also removing the values with passenger count of 0.
train = train.drop(train[train["passenger_count"] == 0 ].index, axis=0)


# In[31]:


train["passenger_count"].describe()


# In[32]:


train["passenger_count"].sort_values(ascending= True)


# In[33]:


#removing passanger_count missing values rows
train = train.drop(train[train['passenger_count'].isnull()].index, axis=0)
train.shape


# In[34]:


train = train.drop(train[train["passenger_count"] == 0.12 ].index, axis=0)
train.shape


# In[35]:


train.isnull().sum()


# In[36]:


train["fare_amount"].sort_values(ascending=False)


# In[37]:


Counter(train["fare_amount"]<0)


# In[38]:


train = train.drop(train[train["fare_amount"]<0].index, axis=0)
train.shape


# In[39]:


##make sure there is no negative values in the fare_amount variable column
train["fare_amount"].min()


# In[40]:


#Also remove the row where fare amount is zero
train = train.drop(train[train["fare_amount"]<1].index, axis=0)
train.shape


# In[41]:



# so we will remove the rows having fare amounting more that 454 as considering them as outliers

train = train.drop(train[train["fare_amount"]> 454 ].index, axis=0)
train.shape


# In[42]:


train = train.drop(train[train['fare_amount'].isnull()].index, axis=0)
train.shape


# In[43]:


train["fare_amount"].describe()


# In[44]:


#Lattitude----(-90 to 90)
#Longitude----(-180 to 180)

# we need to drop the rows having  pickup lattitute and longitute out the range mentioned above

#train = train.drop(train[train['pickup_latitude']<-90])
train[train['pickup_latitude']<-90]
train[train['pickup_latitude']>90]


# In[45]:


#Hence dropping one value of >90
train = train.drop((train[train['pickup_latitude']<-90]).index, axis=0)
train = train.drop((train[train['pickup_latitude']>90]).index, axis=0)


# In[46]:


train[train['pickup_longitude']<-180]
train[train['pickup_longitude']>180]


# In[55]:


train[train['pickup_latitude']<-90]
train[train['pickup_latitude']>90]


# In[47]:


train.shape


# In[48]:


train.isnull().sum()


# In[49]:


test


# In[50]:


# Here pickup_datetime variable is in object so we need to change its data type to datetime
test.pickup_datetime =  pd.to_datetime(test.pickup_datetime,errors='coerce')


# In[51]:


### we will saperate the Pickup_datetime column into separate field like year, month, day of the week, etc

test['year'] = test['pickup_datetime'].dt.year
test['Month'] = test['pickup_datetime'].dt.month
test['Date'] = test['pickup_datetime'].dt.day
test['Day'] = test['pickup_datetime'].dt.dayofweek
test['Hour'] = test['pickup_datetime'].dt.hour
test['Minute'] = test['pickup_datetime'].dt.minute
test['Second'] = test['pickup_datetime'].dt.second


# In[62]:


test.dtypes


# In[52]:


test.isnull().sum()


# In[53]:


#As we know that we have given pickup longitute and latitude values and same for drop. 
#So we need to calculate the distance Using the haversine formula and we will create a new variable called distance
from math import radians, cos, sin, asin, sqrt

def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km
# 1min


# In[54]:


train['distance'] = train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[55]:


train


# In[56]:


test['distance'] = test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[57]:


test


# In[58]:


train.nunique()


# In[59]:


test.nunique()


# In[60]:


##finding decending order of fare to get to know whether the outliers are presented or not
train['distance'].sort_values(ascending=False)


# In[61]:


Counter(train['distance'] == 0)


# In[62]:


Counter(test['distance'] == 0)


# In[63]:


Counter(train['fare_amount'] == 0)


# In[64]:


###we will remove the rows whose distance value is zero

train = train.drop(train[train['distance']== 0].index, axis=0)


# In[65]:


#we will remove the rows whose distance values is very high which is more than 129kms
train = train.drop(train[train['distance'] > 130 ].index, axis=0)
train.shape


# In[84]:


#Now we have splitted the pickup date time variable into different varaibles
#like month, year, day etc so now we dont need to have that pickup_Date variable now.
#Hence we can drop that, Also we have created distance using pickup and drop longitudes and latitudes 
#so we will also drop pickup and drop longitudes and latitudes variables.


drop = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'Minute']


# In[81]:


train.head(4)


# In[92]:


train = train.drop( drop , axis= 1)


# In[93]:


train


# In[99]:


drop_test = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'Minute']
test=test.drop(drop_test,axis=1)
test.head(3)


# In[100]:


test.tail(4)
  


# In[101]:


train.dtypes


# In[102]:


test.dtypes


# In[103]:


train['passenger_count'] = train['passenger_count'].astype('int64')
train['year'] = train['year'].astype('int64')
train['Month'] = train['Month'].astype('int64')
train['Date'] = train['Date'].astype('int64')
train['Day'] = train['Day'].astype('int64')
train['Hour'] = train['Hour'].astype('int64')


# In[104]:


train.dtypes


# In[106]:


#Data Visualization :

#Visualization of following:
#Number of Passengers effects the the fare
#Pickup date and time effects the fare
#Day of the week does effects the fare
#Distance effects the fare


# In[110]:


# Count plot on passenger count
plt.figure(figsize=(15,7))
sns.countplot(x="passenger_count", data=train)


# In[111]:


#Relationship beetween number of passengers and Fare

plt.figure(figsize=(15,7))
plt.scatter(x=train['passenger_count'], y=train['fare_amount'], s=10)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
plt.show()


# In[112]:


#Relationship between date and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=train['Date'], y=train['fare_amount'], s=10)
plt.xlabel('Date')
plt.ylabel('Fare')
plt.show()


# In[113]:


plt.figure(figsize=(15,7))
train.groupby(train["Hour"])['Hour'].count().plot(kind="bar")
plt.show()


# In[114]:


#Relationship between Time and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=train['Hour'], y=train['fare_amount'], s=10)
plt.xlabel('Hour')
plt.ylabel('Fare')
plt.show()


# In[115]:


#impact of Day on the number of cab rides
plt.figure(figsize=(15,7))
sns.countplot(x="Day", data=train)


# In[116]:


#Relationships between day and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=train['Day'], y=train['fare_amount'], s=10)
plt.xlabel('Day')
plt.ylabel('Fare')
plt.show()


# In[117]:


#Relationship between distance and fare 
plt.figure(figsize=(15,7))
plt.scatter(x = train['distance'],y = train['fare_amount'],c = "g")
plt.xlabel('Distance')
plt.ylabel('Fare')
plt.show()


# In[118]:


#Feature Scaling 
#Normality check of training data is uniformly distributed or not-

for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto',color='green')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# In[120]:


#since skewness of target variable is high, apply log transform to reduce the skewness-
train['fare_amount'] = np.log1p(train['fare_amount'])

#since skewness of distance variable is high, apply log transform to reduce the skewness-
train['distance'] = np.log1p(train['distance'])


# In[121]:


#Normality Re-check to check data is uniformly distributed or not after log transformartion

for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto',color='green')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# In[122]:


#Here we can see bell shaped distribution. Hence our continous variables are now normally distributed, 
#we will use not use any Feature Scalling technique. i.e, Normalization or Standarization for our training data


# In[123]:


#Normality check for test data is uniformly distributed or not-

sns.distplot(test['distance'],bins='auto',color='green')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# In[124]:


#since skewness of distance variable is high, apply log transform to reduce the skewness-
test['distance'] = np.log1p(test['distance'])


# In[125]:


#rechecking the distribution for distance
sns.distplot(test['distance'],bins='auto',color='green')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# In[126]:


#Here we can see bell shaped distribution. Hence our continous variables are now normally distributed, 
#we will use not use any Feature Scalling technique. i.e, Normalization or Standarization for our training data


# In[132]:


#Applying ML ALgorithms:
##train test split for further modelling
X_train, X_test, y_train, y_test = train_test_split( train.iloc[:, train.columns != 'fare_amount'], 
                         train.iloc[:, 0], test_size = 0.20, random_state = 1)


# In[133]:


X_train.shape


# In[135]:


X_test.shape


# In[151]:


#Import libraries for LR

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# In[152]:


# Building model on top of training dataset
fit_LR = LinearRegression().fit(X_train , y_train)


# In[160]:


#prediction on train data
pred_train_LR = fit_LR.predict(X_train)


# In[161]:


#prediction on test data
pred_test_LR = fit_LR.predict(X_test)


# In[143]:


# predictions for test model
predictions_LR_test = model.predict(X_test)


# In[165]:




##calculating RMSE for train data
RMSE_train_LR= np.sqrt(mean_squared_error(y_train, pred_train_LR))
RMSE_train_LR


# In[166]:


##calculating RMSE for test data
RMSE_test_LR = np.sqrt(mean_squared_error(y_test, pred_test_LR))
RMSE_test_LR


# In[167]:


#calculate R^2 for train data
r2_score(y_train, pred_train_LR)


# In[168]:


#calculate R^2 for train data
r2_score(y_test, pred_test_LR)


# In[190]:


##Decision tree Model
fit_DT = DecisionTreeRegressor(max_depth = 2).fit(X_train,y_train)


# In[193]:


fit_DT


# In[194]:


#prediction on train data
pred_train_DT = fit_DT.predict(X_train)


# In[174]:


pred_train_DT


# In[195]:


#prediction on test data
pred_test_DT = fit_DT.predict(X_test)


# In[198]:


pred_test_DT 


# In[199]:


##calculating RMSE for train data
RMSE_train_DT = np.sqrt(mean_squared_error(y_train, pred_train_DT))

##calculating RMSE for test data
RMSE_test_DT = np.sqrt(mean_squared_error(y_test, pred_test_DT))


# In[200]:


RMSE_train_DT


# In[201]:


RMSE_test_DT


# In[202]:


## R^2 calculation for train data
r2_score(y_train, pred_train_DT)


# In[203]:


## R^2 calculation for train data
r2_score(y_test, pred_test_DT)


# In[206]:


#Random Forest Model 
fit_RF = RandomForestRegressor(n_estimators = 200).fit(X_train,y_train)


# In[211]:


#prediction on train data
pred_train_RF = fit_RF.predict(X_train)
pred_train_RF


# In[213]:


#prediction on test data
pred_test_RF = fit_RF.predict(X_test)


# In[214]:


##calculating RMSE for train data
RMSE_train_RF = np.sqrt(mean_squared_error(y_train, pred_train_RF))
##calculating RMSE for test data
RMSE_test_RF = np.sqrt(mean_squared_error(y_test, pred_test_RF))


# In[215]:


print("Root Mean Squared Error For Training data = "+str(RMSE_train_RF))
print("Root Mean Squared Error For Test data = "+str(RMSE_test_RF))


# In[216]:


## calculate R^2 for train data

r2_score(y_train, pred_train_RF)


# In[217]:


#calculate R^2 for test data
r2_score(y_test, pred_test_RF)


# In[222]:


pred_test_RF


# In[228]:



from sklearn.model_selection import GridSearchCV


# In[263]:


## Grid Search CV for random Forest model
regr = RandomForestRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))
#Create the grid
grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_rf = GridSearchCV(regr, param_grid = grid_search, cv = 5)

gridcv_rf = gridcv_rf.fit(X_train,y_train)


# In[264]:


#Apply model on test data
predictions_GRF_test_Df = gridcv_rf.predict(test)


# In[262]:


predictions_GRF_test_Df


# In[259]:


test['Predicted_fare'] = predictions_GRF_test_Df


# In[260]:


test.head(10)


# In[266]:


test.to_csv('test2.csv')


# In[251]:





# In[ ]:





# In[ ]:




