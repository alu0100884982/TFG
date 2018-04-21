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
import re
import datetime


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
X_train = travel_time_dataframe.iloc[:, 0:14]
y_train = travel_time_dataframe.iloc[:,14]

try:
    conn = psycopg2.connect("dbname='tfgtest1' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""select *
from  a_2_9_40;""")
rows = cur.fetchall()
colnames = [ 'type_day','twenty_min_previous','forty_min_previous','sixty_min_previous','eighty_min_previous','onehundred_min_previous','onehundredtwenty_min_previous','pressure','sea_pressure','wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation','avg_travel_time']
travel_time_dataframe = pd.DataFrame(rows, columns=colnames)
set_option( 'display.width' , 100)
X_test = travel_time_dataframe.iloc[:, 0:14]

try:
    conn = psycopg2.connect("dbname='tfgtraining2' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""select avg_travel_time
from travel_time_intersection_to_tollgate_training2
where intersection_id = 'A' AND tollgate_id = 2 AND time_window[1].time = TIME '09:40:00'
order by intersection_id, tollgate_id, time_window;""")
rows = cur.fetchall()
colnames = [ 'avg_travel_time']
travel_time_dataframe = pd.DataFrame(rows, columns=colnames)
y_test = travel_time_dataframe.iloc[:, 0]
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

###########################################################################################################################
cur = conn.cursor()
cur.execute("""SELECT * FROM travel_time_intersection_to_tollgate_training2 WHERE (time_window[1].time BETWEEN TIME '08:00:00' AND TIME '09:40:00') OR (time_window[1].time BETWEEN TIME '17:00:00' AND TIME '18:40:00') ORDER BY intersection_id, tollgate_id, time_window """)
rows = cur.fetchall()
colnames = ['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time']
intervals_to_predict_real_avgtraveltime = pd.DataFrame(rows, columns=colnames)
routes = np.array(intervals_to_predict_real_avgtraveltime.iloc[:,0:2].values.tolist())
time_intervals = np.array(intervals_to_predict_real_avgtraveltime.iloc[:,2].values.tolist())
routes = np.unique(routes, axis=0);
aux = np.array([])
for time_interval in time_intervals:
        time_interval[0] = datetime.datetime(2016, 10, 18, time_interval[0].hour, time_interval[0].minute)
        aux = np.append(aux,time_interval[0]);
time_intervals = sorted(set(aux))
print(time_intervals)
real_avgtraveltime_values = np.array([])
predicted_avgtraveltime_values = np.array([])
colnames = [ 'type_day','twenty_min_previous','forty_min_previous','sixty_min_previous','eighty_min_previous','onehundred_min_previous','onehundredtwenty_min_previous','pressure','sea_pressure','wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation','avg_travel_time']


suma = 0;
cuenta = 0;
errores_predicciones_intervalos = {
		"XGBoost" : 0,
		"Linear Regression" : 0,
		"LightGBM" : 0,
		"MLP" : 0,
		"SVM" : 0,
		"KNN" : 0
	}
	
errores_predicciones_rutas = {
		"XGBoost" : 0,
		"Linear Regression" : 0,
		"LightGBM" : 0,
		"MLP" : 0,
		"SVM" : 0,
		"KNN" : 0
	}
	
with open('predicciones.txt', 'a') as the_file:
        for route in routes:
             suma_intervalos_tiempo = 0;
             for interval in time_intervals:
                print("RUTA ", str(route) + " INTERVAL ", str(interval))
                the_file.write(str(route[0]) + "," + str(route[1]) + "," + str(interval) + "\n")
                colnames =       [ 'type_day','twenty_min_previous','forty_min_previous','sixty_min_previous','eighty_min_previous','onehundred_min_previous','onehundredtwenty_min_previous','pressure','sea_pressure','wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation','avg_travel_time']          
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
               
                try:
                    conn = psycopg2.connect("dbname='tfgtraining2' user='javisunami' host='localhost' password='javier123'")
                except:
                    print("I am unable to connect to the database")

                cur = conn.cursor()
                query = "select time_window[1].date, avg_travel_time from travel_time_intersection_to_tollgate_training2 where intersection_id = '" + str(route[0]) +"' AND tollgate_id = " + str(route[1]) +" AND time_window[1].time = TIME '"+ str(interval.hour) +":"+str(interval.minute)  + ":00' order by intersection_id, tollgate_id, time_window;"
                cur.execute(query)
                rows = cur.fetchall()
                colnames = [ 'date', 'avg_travel_time']
                travel_time_dataframe = pd.DataFrame(rows, columns=colnames)
                the_file.write("Valores reales, "+ str(','.join(map(str, travel_time_dataframe['avg_travel_time'].tolist()))) + "\n")
                model = XGBRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_test_sum = 0
                for index, fila in travel_time_dataframe.iterrows():
                        y_test_sum += abs((fila['avg_travel_time'] - y_pred[fila['date'].day - 18]) / fila['avg_travel_time'])
                y_test_sum /= len(travel_time_dataframe);
                errores_predicciones_intervalos["XGBoost"] += y_test_sum
                model = XGBRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                the_file.write("XGBoost,"+ str(','.join(map(str, y_pred))) + "\n")
                y_test_sum = 0
                for index, fila in travel_time_dataframe.iterrows():
                        y_test_sum += abs((fila['avg_travel_time'] - y_pred[fila['date'].day - 18]) / fila['avg_travel_time'])
                y_test_sum /= len(travel_time_dataframe);
                model = linear_model.LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                the_file.write("Regresión Lineal, "+  str(','.join(map(str, y_pred))) + "\n")
                errores_predicciones_intervalos["Linear Regression"] += y_test_sum
               

                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                mlp = MLPRegressor(hidden_layer_sizes=(13,13,13),max_iter=10000)
                mlp.fit(X_train,y_train)
                y_pred = mlp.predict(X_test)
                the_file.write("Red Neuronal, "+  str(','.join(map(str, y_pred))) + "\n")
                y_test_sum = 0
                for index, fila in travel_time_dataframe.iterrows():
                        y_test_sum += abs((fila['avg_travel_time'] - y_pred[fila['date'].day - 18]) / fila['avg_travel_time'])
                y_test_sum /= len(travel_time_dataframe);
                errores_predicciones_intervalos["MLP"] += y_test_sum

                model = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
                    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
                model.fit(X_train, y_train) 
                y_pred = model.predict(X_test)
                the_file.write("Máquina de Soporte Vectorial,: "+ str(','.join(map(str, y_pred))) + "\n")
                y_test_sum = 0
                for index, fila in travel_time_dataframe.iterrows():
                        y_test_sum += abs((fila['avg_travel_time'] - y_pred[fila['date'].day - 18]) / fila['avg_travel_time'])
                y_test_sum /= len(travel_time_dataframe);
                errores_predicciones_intervalos["SVM"] += y_test_sum
               
                model =KNeighborsRegressor(n_neighbors=3)
                model.fit(X_train, y_train) 
                y_pred = model.predict(X_test)
                the_file.write("KNN, "+  str(','.join(map(str, y_pred))) + "\n\n")
                y_test_sum = 0
                for index, fila in travel_time_dataframe.iterrows():
                        y_test_sum += abs((fila['avg_travel_time'] - y_pred[fila['date'].day - 18]) / fila['avg_travel_time'])
                y_test_sum /= len(travel_time_dataframe);
                errores_predicciones_intervalos["KNN"] += y_test_sum
                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
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
                the_file.write("LightGBM,: "+ str(','.join(map(str, y_pred))) + "\n")
             for key, value in errores_predicciones_rutas.items():
                errores_predicciones_rutas[key] += errores_predicciones_intervalos[key]/len(time_intervals)
                errores_predicciones_intervalos[key] = 0  
        for key, value in errores_predicciones_rutas.items():
                errores_predicciones_rutas[key] = errores_predicciones_rutas[key]/len(routes);
        print ("Error : ", errores_predicciones_rutas);      

