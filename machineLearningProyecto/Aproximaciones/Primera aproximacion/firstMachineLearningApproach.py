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
error5 = sum(y_pred - y_test)

print("Error XGBoost: ", error1);
print("Error Linear Regression: ", error2);
print("Error LightGBM: ", error3);
print("Error Multiple Layer Perception: ", error4);
print("Error SVM: ", error5);
print("y_pred : ", y_pred)
print("y_test :", y_test);

MAPE = 0
for i in range(0,len(y_pred)):
        MAPE += abs((y_test.iloc[i] - y_pred[i])/y_test.iloc[i])
MAPE /= len(y_pred);
print("MAPE : ", MAPE)
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
real_avgtraveltime_values = intervals_to_predict_real_avgtraveltime.iloc[:,3]
routes = np.array(intervals_to_predict_real_avgtraveltime.iloc[:,0:2].values.tolist())
print(routes);


