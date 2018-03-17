import psycopg2
import pandas as pd
from pandas import set_option
from numpy import loadtxt
from xgboost.sklearn import XGBRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import math
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor


try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""select *
from  a_2_9_40;""")
rows = cur.fetchall()
colnames = [ 'type_day','twenty_min_previous','forty_min_previous','sixty_min_previous','eighty_min_previous','onehundred_min_previous','onehundredtwenty_min_previous','pressure','sea_pressure','wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation','avg_travel_time']
travel_time_dataframe = pd.DataFrame(rows, columns=colnames)
set_option( 'display.width' , 100)
X = travel_time_dataframe.iloc[:, 0:14]
Y = travel_time_dataframe.iloc[:,14]
seed = 14
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
error1 = mean_squared_error(y_pred, y_test)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
error2 = mean_squared_error(y_pred, y_test)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
error3 = mean_squared_error(y_pred, y_test)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPRegressor(hidden_layer_sizes=(13,13,13),max_iter=10000)
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)
error4 = mean_squared_error(y_pred, y_test)

model = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)
error5 = mean_squared_error(y_pred, y_test)
print("y_pred : ", y_pred)
print("y_test: ", y_test)
model =KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)
error6 = mean_squared_error(y_pred, y_test)
print("Mean Squared Error XGBoost: ", error1)
print("Mean Squared Error Linear Regression: ", error2)
print("Mean Squared Error LightGBM: ", error3)
print("Mean Squared Error Multiple Layer Perception: ", error4)
print("Mean Squared Error SVM: ", error5)
print("Mean Squared Error KNN: ", error6)

