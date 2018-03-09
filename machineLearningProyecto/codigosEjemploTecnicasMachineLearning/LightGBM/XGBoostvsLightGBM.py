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

print("Clases : " + str(l.classes_))
'''
Series is a one-dimensional labeled array capable of holding data of any type (integer, string, float, python objects, etc.).
 The axis labels are collectively called index.
'''
data.Income=Series(l.transform(data.Income))  #label encoding our target variable 
'''
#Returns object containing counts of unique values. 
The resulting object will be in descending order so that the first element is the most frequently-occurring element
'''
pd.set_option('display.width', 200)
print(data.Income.value_counts()) 

 
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

'''
axis : int or axis name

    Whether to drop labels from the index (0 / ‘index’) or columns (1 / ‘columns’).

'''
#removing categorical features 
data.drop(['workclass','education','marital_Status','occupation','relationship','race','sex','native_country'],axis=1,inplace=True) 

#Merging one hot encoded features with our dataset 'data' 
data=pd.concat([data,one_hot_workclass,one_hot_education,one_hot_marital_Status,one_hot_occupation,one_hot_relationship,one_hot_race,one_hot_sex,one_hot_native_country],axis=1) 


'''
numpy.unique

    Find the unique elements of an array.
'''
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


#The data is stored in a DMatrix object 
#label is used to define our outcome variable
dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test)

#setting parameters for xgboost
parameters={'max_depth':7, 'eta':1, 'silent':1,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05}

#training our model 
num_round=50
from datetime import datetime 
start = datetime.now() 
xg=xgb.train(parameters,dtrain,num_round) 
stop = datetime.now()

#Execution time of the model 
execution_time_xgb = stop-start 
print("Execution time xgb : " + str(execution_time_xgb))

#datetime.timedelta( , , ) representation => (days , seconds , microseconds) 
#now predicting our model on test set 
ypred=xg.predict(dtest) 


#Converting probabilities into 1 or 0  
for i in range(0,9769): 
    if ypred[i]>=.5:       # setting threshold to .5 
       ypred[i]=1 
    else: 
       ypred[i]=0  

#calculating accuracy of our model 
from sklearn.metrics import accuracy_score 
accuracy_xgb = accuracy_score(y_test,ypred) 
print("Accuracy xgb : " + str(accuracy_xgb))

# Light GBM

train_data=lgb.Dataset(x_train,label=y_train)

#setting parameters for lightgbm
param = {'num_leaves':150, 'objective':'binary','max_depth':7,'learning_rate':.05,'max_bin':200}
param['metric'] = ['auc', 'binary_logloss']

#Here we have set max_depth in xgb and LightGBM to 7 to have a fair comparison between the two.

#training our model using light gbm
num_round=50
start=datetime.now()
lgbm=lgb.train(param,train_data,num_round)
stop=datetime.now()

#Execution time of the model
execution_time_lgbm = stop-start
execution_time_lgbm

#predicting on test set
ypred2=lgbm.predict(x_test)
ypred2[0:5]  # showing first 5 predictions

#converting probabilities into 0 or 1
for i in range(0,9769):
    if ypred2[i]>=.5:       # setting threshold to .5
       ypred2[i]=1
    else:  
       ypred2[i]=0

#calculating accuracy
accuracy_lgbm = accuracy_score(ypred2,y_test)
accuracy_lgbm
y_test.value_counts()

from sklearn.metrics import roc_auc_score

#calculating roc_auc_score for xgboost
auc_xgb =  roc_auc_score(y_test,ypred)
auc_xgb

#calculating roc_auc_score for light gbm. 
auc_lgbm = roc_auc_score(y_test,ypred2)
comparison_dict = {'accuracy score':(accuracy_lgbm,accuracy_xgb),'auc score':(auc_lgbm,auc_xgb),'execution time':(execution_time_lgbm,execution_time_xgb)}

#Creating a dataframe 'comparison_df' for comparing the performance of Lightgbm and xgb
comparison_df = DataFrame(comparison_dict) 
comparison_df.index= ['LightGBM','xgboost'] 
print(comparison_df)

