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
import datetime
import re

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

print("Error XGBoost: ", error1);
print("Error Linear Regression: ", error2);
print("Error LightGBM: ", error3);
print("Error Multiple Layer Perception: ", error4);
print("Error SVM: ", error5);
print("y_pred : ", y_pred)
print("y_test :", y_test);



#####################################################################################################################################################################################################
cur.execute("""CREATE EXTENSION dblink""")
cur.execute("""SELECT P.*
FROM dblink('dbname=tfgtraining2 port=5432 host=127.0.0.1 user=javisunami password=javier123', 'SELECT * FROM travel_time_intersection_to_tollgate_training2 WHERE (time_window[1].time BETWEEN TIME ''08:00:00'' AND TIME ''09:40:00'') OR (time_window[1].time BETWEEN TIME ''17:00:00'' AND TIME ''18:40:00'') ORDER BY intersection_id, tollgate_id, time_window') 
AS P( intersection_id char(1),
tollgate_id smallint,
time_window varchar(50),
avg_travel_time float);""")
rows = cur.fetchall()
colnames = ['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time']
intervals_to_predict_real_avgtraveltime = pd.DataFrame(rows, columns=colnames)
routes = np.array(intervals_to_predict_real_avgtraveltime.iloc[:,0:2].values.tolist())
time_intervals = np.array(intervals_to_predict_real_avgtraveltime.iloc[:,2].values.tolist())
routes = np.unique(routes, axis=0);
aux = np.array([])
for i,time_interval in enumerate(time_intervals):
        time_interval = re.sub(r'[{\"}]', '', time_interval).split(',')
        time_interval[0] = datetime.datetime.strptime(time_interval[0] , '%Y-%m-%d %H:%M:%S')
        time_interval[1] = datetime.datetime.strptime(time_interval[1] , '%Y-%m-%d %H:%M:%S')
        aux = np.append(aux,(datetime.time(time_interval[0].hour,time_interval[0].minute), datetime.time(time_interval[1].hour,time_interval[1].minute)));
time_intervals = np.unique(aux);
time_intervals = np.delete(time_intervals, -1)
time_intervals = np.delete(time_intervals, 6)
print("TIME_INTERVALS: ", time_intervals);
real_avgtraveltime_values = np.array([])
predicted_avgtraveltime_values = np.array([])
colnames = [ 'type_day','twenty_min_previous','forty_min_previous','sixty_min_previous','eighty_min_previous','onehundred_min_previous','onehundredtwenty_min_previous','pressure','sea_pressure','wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation','avg_travel_time']

suma_rutas = 0;
for route in routes:
     suma_intervalos_tiempo = 0;
     for interval in time_intervals:
        conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
        cur = conn.cursor()
        query = "select * from " + route[0].lower() + "_" + str(route[1]) + "_" + str(interval.hour) + "_" + str(interval.minute) + ";"
        cur.execute(query)
        rows = cur.fetchall()
        dataframe_traveltime = pd.DataFrame(rows, columns=colnames)
        X_train = dataframe_traveltime.iloc[:, 0:14]
        y_train = dataframe_traveltime.iloc[:, 14]
        
        conn = psycopg2.connect("dbname='tfgtest1' user='javisunami' host='localhost' password='javier123'")
        cur = conn.cursor()
        query = "select * from " + route[0].lower() + "_" + str(route[1]) + "_" + str(interval.hour) + "_" + str(interval.minute) + ";"
        cur.execute(query)
        rows = cur.fetchall()
        dataframe_traveltime = pd.DataFrame(rows, columns=colnames)
        X_test = dataframe_traveltime.iloc[:, 0:14]
        y_test = dataframe_traveltime.iloc[:, 14]
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
        train_data=lgb.Dataset(X_train,label=y_train)
        lgbm=lgb.train(params,train_data,50);
        predicted_values = lgbm.predict(X_test)
        print ("predicted values : ", predicted_values)
        real_values = intervals_to_predict_real_avgtraveltime[(intervals_to_predict_real_avgtraveltime.intersection_id == route[0]) & (intervals_to_predict_real_avgtraveltime.tollgate_id == route[1]) ];
              #suma_intervalos_tiempo += abs((real_value - predicted_value) / real_value);
     #suma_rutas += suma_intervalos_tiempo / time_intervals.size;
#print (suma_rutas / routes.size);   
               

