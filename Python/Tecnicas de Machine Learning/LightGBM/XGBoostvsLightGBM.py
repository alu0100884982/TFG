#importing standard libraries 
import numpy as np 
import pandas as pd 
from pandas import Series, DataFrame 

#import lightgbm and xgboost 
import lightgbm as lgb 
import xgboost as xgb 

#loading our training dataset 'adult.csv' with name 'data' using pandas 
data=pd.read_csv('adult.csv',header=None) 

#Assigning names to the columns 
data.columns=['age','workclass','fnlwgt','education','education-num','marital_Status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','Income'] 

#glimpse of the dataset 
data.head() 

# Label Encoding our target variable 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

'''
LabelEncoder is a utility class to help normalize labels such that they contain only values between 0 and n_classes-1. This is sometimes useful for writing efficient Cython routines. LabelEncoder can be used as follows:
>>>

>>> from sklearn import preprocessing
>>> le = preprocessing.LabelEncoder()
>>> le.fit([1, 2, 2, 6])
LabelEncoder()
>>> le.classes_
array([1, 2, 6])
>>> le.transform([1, 1, 2, 6]) -> Transform labels to normalized encoding.
array([0, 0, 1, 2])
>>> le.inverse_transform([0, 0, 1, 2])
array([1, 1, 2, 6])
'''

l=LabelEncoder() 
l.fit(data.Income) 

l.classes_ 
'''
Series is a one-dimensional labeled array capable of holding data of any type (integer, string, float, python objects, etc.).
 The axis labels are collectively called index.
'''
data.Income=Series(l.transform(data.Income))  #label encoding our target variable 
'''
#Returns object containing counts of unique values. 
The resulting object will be in descending order so that the first element is the most frequently-occurring element
'''
data.Income.value_counts() 

 
'''
>>> import pandas as pd
In [33]: p.get_dummies(['H','G','L'])
Out[33]: 
   G  H  L
0  0  1  0
1  1  0  0
2  0  0  1

'''
#One Hot Encoding of the Categorical features 
one_hot_workclass=pd.get_dummies(data.workclass) 
one_hot_education=pd.get_dummies(data.education) 
one_hot_marital_Status=pd.get_dummies(data.marital_Status) 
one_hot_occupation=pd.get_dummies(data.occupation)
one_hot_relationship=pd.get_dummies(data.relationship) 
one_hot_race=pd.get_dummies(data.race) 
one_hot_sex=pd.get_dummies(data.sex) 
one_hot_native_country=pd.get_dummies(data.native_country) 

#removing categorical features 
data.drop(['workclass','education','marital_Status','occupation','relationship','race','sex','native_country'],axis=1,inplace=True) 

 

#Merging one hot encoded features with our dataset 'data' 
data=pd.concat([data,one_hot_workclass,one_hot_education,one_hot_marital_Status,one_hot_occupation,one_hot_relationship,one_hot_race,one_hot_sex,one_hot_native_country],axis=1) 

#removing dulpicate columns 
 _, i = np.unique(data.columns, return_index=True) 
data=data.iloc[:, i] 

#Here our target variable is 'Income' with values as 1 or 0.  
#Separating our data into features dataset x and our target dataset y 
x=data.drop('Income',axis=1) 
y=data.Income 

 

#Imputing missing values in our target variable 
y.fillna(y.mode()[0],inplace=True) 

#Now splitting our dataset into test and train 
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)
