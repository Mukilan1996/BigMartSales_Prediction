
"""
Created on Thu Sep  4 14:52:32 2020

@author: Mukilan


"""

""" Using different Alogorithmn to find the best algorithmn for regression Problem statement. 
    I used algorithmn like Linear, Random Forest, Gradient Boost Regression. And I visualize the data using seaborn."""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

train=pd.read_csv("train.csv")
train.shape


train.groupby(['Item_Identifier'])['Item_Weight'].mean()
train['Item_Weight']=train.groupby('Item_Identifier')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))

visibility_avg = train.pivot_table(values='Item_Visibility', index='Item_Identifier')

#Replacing null values in mean values
missing_values = (train['Item_Visibility'] == 0)

print ('Number of 0 values initially: %d'%sum(missing_values))
train.loc[missing_values,'Item_Visibility'] = train.loc[missing_values,'Item_Identifier'].apply(lambda x: visibility_avg.at[x, 'Item_Visibility'])
print ('Number of 0 values after modification: %d'%sum(train['Item_Visibility'] == 0))


outlet_size_mode = train.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x.astype('str')).mode[0]))
print ('Mode for each Outlet_Type:')
print (outlet_size_mode)


missing_values = train['Outlet_Size'].isnull() 

print ('\nOrignal #missing: %d'% sum(missing_values))
train.loc[missing_values,'Outlet_Size'] = train.loc[missing_values,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print (sum(train['Outlet_Size'].isnull()))

train['Item_Type_Combined'] = train['Item_Identifier'].apply(lambda x: x[0:2])
train['Item_Type_Combined'] = train['Item_Type_Combined'].replace({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
train['Item_Type_Combined'].value_counts()


print('Original Categories:')
print(train['Item_Fat_Content'].value_counts())

print('\nModified Categories:')
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print(train['Item_Fat_Content'].value_counts())



train=train.dropna(axis=0,how='any')

train.to_csv('bigtrain.csv',index=False)

#distribution plot
sns.distplot(train.Item_Outlet_Sales, bins = 25)
plt.xlabel("Item_Outlet_Sales")
plt.ylabel("Number of Sales")
plt.title("Item_Outlet_Sales Distribution")

print ("Skew is:", train.Item_Outlet_Sales.skew())
print("Kurtosis: %f" % train.Item_Outlet_Sales.kurt())

numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes

corr =numeric_features.corr()
corr

f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True);


sns.countplot(train.Item_Fat_Content)


sns.countplot(train.Item_Type)
plt.xticks(rotation=90)


sns.countplot(train.Outlet_Size)


sns.countplot(train.Outlet_Location_Type)

sns.countplot(train.Outlet_Type)
plt.xticks(rotation=90)

plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
plt.plot(train.Item_Weight, train["Item_Outlet_Sales"])


plt.figure(figsize=(12,7))

Outlet_Establishment_Year_pivot = \
train.pivot_table(index='Outlet_Establishment_Year', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Establishment_Year_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Establishment_Year")
plt.ylabel("Sqrt Item_Outlet_Sales")
plt.title("Impact of Outlet_Establishment_Year on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


Item_Fat_Content_pivot = \
train.pivot_table(index='Item_Fat_Content', values="Item_Outlet_Sales", aggfunc=np.median)
Item_Fat_Content_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Item_Fat_Content")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Item_Fat_Content on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()

Outlet_Identifier_pivot = \
train.pivot_table(index='Outlet_Identifier', values='Item_Outlet_Sales', aggfunc=np.median)
Outlet_Identifier_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel('Outlet_Identifier ')
plt.ylabel('Item_Outlet_Sales')
plt.title('Impact of Outlet_Identifier on Item_Outlet_Sales')
plt.xticks(rotation=0)
plt.show()



Outlet_Type_pivot = \
train.pivot_table(index='Outlet_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Type_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Type ")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()



Outlet_Location_Type_pivot = \
train.pivot_table(index='Outlet_Location_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Location_Type_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Location_Type ")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Location_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()





train = pd.get_dummies(train)

y = train['Item_Outlet_Sales']
x = train.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'],axis=1)
x=pd.get_dummies(x)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=123)

''''linear regression model'''''

regressor = LinearRegression()
regressor.fit(x,y)
#regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


print(round(r2_score(y_test,y_pred),2))                        '''      predicted r score : 0.57'''

'''''   random forest '''''

regr = RandomForestRegressor(n_estimators=1000, random_state=30)
regr.fit(x_train,y_train)
#regressor.fit(x_train,y_train)
y_pred = regr.predict(x_test)

round(r2_score(y_test,y_pred),2)                               ''''  predicted r score : 0.58''''


'''''    Gradient Boost Regression '''''


params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}

reg = sklearn.ensemble.GradientBoostingRegressor(**params)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

round(r2_score(y_test,y_pred),2)                              '''' predicted r score : 0.65''''









""" I test with test.csv to check algorithm performance and accuracy"""


test=pd.read_csv("test.csv")
test.shape


test.groupby(['Item_Identifier'])['Item_Weight'].mean()
test['Item_Weight']=test.groupby('Item_Identifier')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))

visibility_avg = test.pivot_table(values='Item_Visibility', index='Item_Identifier')


missing_values = (test['Item_Visibility'] == 0)

print ('Number of 0 values initially: %d'%sum(missing_values))
test.loc[missing_values,'Item_Visibility'] = test.loc[missing_values,'Item_Identifier'].apply(lambda x: visibility_avg.at[x, 'Item_Visibility'])
print ('Number of 0 values after modification: %d'%sum(test['Item_Visibility'] == 0))



outlet_size_mode = test.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x.astype('str')).mode[0]))
print ('Mode for each Outlet_Type:')
print (outlet_size_mode)

missing_values = test['Outlet_Size'].isnull() 


print ('\nOrignal #missing: %d'% sum(missing_values))
test.loc[missing_values,'Outlet_Size'] = test.loc[missing_values,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print (sum(test['Outlet_Size'].isnull()))

test['Item_Type_Combined'] = test['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
test['Item_Type_Combined'] = test['Item_Type_Combined'].replace({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
test['Item_Type_Combined'].value_counts()



print('Original Categories:')
print(test['Item_Fat_Content'].value_counts())

print('\nModified Categories:')
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print(test['Item_Fat_Content'].value_counts())

test=test.dropna()

test = pd.get_dummies(test)

Item_Identifier = pd.DataFrame(test['Item_Identifier'],columns=['Item_Identifier'])
Outlet_Identifier=pd.DataFrame(test['Outlet_Identifier'],columns=['Outlet_Identifier'])
test = test.drop(['Item_Identifier','Outlet_Identifier'],axis=1)

y_test_pred = regressor.predict(test)

Item_Outlet_Sales_Predicted = pd.DataFrame(y_test_pred,columns=["Item_Outlet_Sales_Predicted"])

finalData = pd.concat([Item_Identifier,Outlet_Identifier,Item_Outlet_Sales_Predicted],axis=1)

finalData.dropna()



""""""""" Finally I conclude:

          For the Bigmart Sales Prediction the  Gradient Boost Regression Algorithmn is good
          Because is high score compare to all algorithmn
          """""""""""















