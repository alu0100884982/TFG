import psycopg2
import pandas as pd
from pandas import set_option
from numpy import loadtxt
from xgboost.sklearn import XGBRegressor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import math
import matplotlib.pyplot as plt
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
import math 

'''
try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""select *
from  a_2_9_40;""")
rows = cur.fetchall()
colnames = [ 'type_day','twenty_min_previous','forty_min_previous','sixty_min_previous','eighty_min_previous','onehundred_min_previous','onehundredtwenty_min_previous','pressure','sea_pressure','wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation','volume']
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
colnames = [ 'type_day','twenty_min_previous','forty_min_previous','sixty_min_previous','eighty_min_previous','onehundred_min_previous','onehundredtwenty_min_previous','pressure','sea_pressure','wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation','volume']
travel_time_dataframe = pd.DataFrame(rows, columns=colnames)
set_option( 'display.width' , 100)
X_test = travel_time_dataframe.iloc[:, 0:14]

try:
    conn = psycopg2.connect("dbname='tfgtraining2' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""select volume
from travel_time_intersection_to_tollgate_training2
where intersection_id = 'A' AND tollgate_id = 2 AND time_window[1].time = TIME '09:40:00'
order by intersection_id, tollgate_id, time_window;""")
rows = cur.fetchall()
colnames = [ 'volume']
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
'''
###########################################################################################################################
try:
    conn = psycopg2.connect("dbname='tfgtraining2' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")
cur = conn.cursor()
cur.execute("""SELECT tollgate_id, direction, time_window, volume FROM traffic_volume_tollgates_training2 WHERE (time_window[1].time BETWEEN TIME '08:00:00' AND TIME '09:40:00') OR (time_window[1].time BETWEEN TIME '17:00:00' AND TIME '18:40:00') ORDER BY tollgate_id, direction, time_window """)
rows = cur.fetchall()
colnames = ['tollgate_id','direction' ,'time_window', 'volume']
intervals_to_predict_real_volume = pd.DataFrame(rows, columns=colnames)
pairs = np.array(intervals_to_predict_real_volume.iloc[:,0:2].values.tolist())
time_intervals = np.array(intervals_to_predict_real_volume.iloc[:,2].values.tolist())
pairs = np.unique(pairs, axis=0);
aux = np.array([])
for time_interval in time_intervals:
        time_interval[0] = datetime.datetime(2016, 10, 18, time_interval[0].hour, time_interval[0].minute)
        aux = np.append(aux,time_interval[0]);
time_intervals = sorted(set(aux))
print(time_intervals)
real_volume_values = np.array([])
predicted_volume_values = np.array([])
colnames = [ 'type_day','twenty_min_previous','forty_min_previous','sixty_min_previous','eighty_min_previous','onehundred_min_previous','onehundredtwenty_min_previous','pressure','sea_pressure','wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation','volume']


suma = 0;
cuenta = 0;
errores_predicciones_intervalos = {
		"XGBoost" : 0,
		"Linear Regression" : 0,
		"LightGBM" : 0,
		"MLP" : 0,
		"SVR" : 0,
		"KNN" : 0
	}
	
errores_predicciones_rutas = {
		"XGBoost" : 0,
		"Linear Regression" : 0,
		"LightGBM" : 0,
		"MLP" : 0,
		"SVR" : 0,
		"KNN" : 0,
		"LightGBM":0
	}

valores_predichos = {}
	
with open('predicciones.txt', 'a') as the_file:
        for pair in pairs:
             suma_intervalos_tiempo = 0;
             for interval in time_intervals:
                print("PAIR ", str(pair) + " INTERVAL ", str(interval.time()) + " - " + str((interval + datetime.timedelta(minutes = 20)).time()))
                the_file.write(str(pair[0]) + "," + str(pair[1]) + "," + str(interval.time()) + " - " + str((interval + datetime.timedelta(minutes = 20)).time()) + "\n")
                colnames =       [ 'type_day','twenty_min_previous','forty_min_previous','sixty_min_previous','eighty_min_previous','onehundred_min_previous','onehundredtwenty_min_previous','pressure','sea_pressure','wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation','volume']          
                conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
                cur = conn.cursor()
                query = "select * from volume_" + str(pair[0]) + "_" + str(pair[1]) + "_" + str(interval.hour) + "_" + str(interval.minute) + ";"
                cur.execute(query)
                rows = cur.fetchall()
                dataframe_traveltime = pd.DataFrame(rows, columns=colnames)
                X_train = dataframe_traveltime.iloc[:, 0:14]
                y_train = dataframe_traveltime.iloc[:, 14]
                
                conn = psycopg2.connect("dbname='tfgtest1' user='javisunami' host='localhost' password='javier123'")
                cur = conn.cursor()
                query = "select * from volume_" + str(pair[0]) + "_" + str(pair[1]) + "_" + str(interval.hour) + "_" + str(interval.minute) + ";"
                cur.execute(query)
                rows = cur.fetchall()
                dataframe_traveltime = pd.DataFrame(rows, columns=colnames)
                X_test = dataframe_traveltime.iloc[:, 0:14]
               
                try:
                    conn = psycopg2.connect("dbname='tfgtraining2' user='javisunami' host='localhost' password='javier123'")
                except:
                    print("I am unable to connect to the database")

                cur = conn.cursor()
                query = "select time_window[1].date, volume from traffic_volume_tollgates_training2 where tollgate_id = '" + str(pair[0]) +"' AND direction = " + str(pair[1]) +" AND time_window[1].time = TIME '"+ str(interval.hour) +":"+str(interval.minute)  + ":00' order by tollgate_id, direction, time_window;"
                cur.execute(query)
                rows = cur.fetchall()
                colnames = [ 'date', 'volume']
                travel_time_dataframe = pd.DataFrame(rows, columns=colnames)
                the_file.write("Valores reales, "+ str(','.join(map(str, travel_time_dataframe['volume'].tolist()))) + "\n")
                
                #XGBoost
                model = XGBRegressor(objective='reg:linear', max_depth = 6, subsample = 0.8, colsample_bytree = 0.8, silent = 1, eval_metric = 'rmse', booster = 'gbtree')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                predicciones=[]
                y_test_sum = 0
                for index, fila in travel_time_dataframe.iterrows():
                        valores_predichos[(pair[0],pair[1],fila['date'].day,interval.strftime("%H:%M"))] = [fila['volume'],math.ceil(y_pred[fila['date'].day - 18])]
                        predicciones.append(y_pred[fila['date'].day - 18])
                        y_test_sum += abs((fila['volume'] - math.ceil(y_pred[fila['date'].day - 18])) / fila['volume'])
                y_test_sum /= len(travel_time_dataframe);
                errores_predicciones_intervalos["XGBoost"] += y_test_sum
                the_file.write("XGBoost,"+ str(','.join(map(str, predicciones))) + "\n")
                
                #Linear Regression
                model = sm.OLS(y_train, X_train).fit_regularized()
                y_pred = model.predict(X_test)
                print("PREDICCIONES : ", y_pred)
                y_test_sum = 0
                predicciones=[]
                for index, fila in travel_time_dataframe.iterrows():
                        valores_predichos[(pair[0],pair[1],fila['date'].day,interval.strftime("%H:%M"))].append(math.ceil(y_pred[fila['date'].day - 18]))
                        predicciones.append(y_pred[fila['date'].day - 18])
                        y_test_sum += abs((fila['volume'] - math.ceil(y_pred[fila['date'].day - 18])) / fila['volume'])
                y_test_sum /= len(travel_time_dataframe);
                the_file.write("Regresión Lineal, "+  str(','.join(map(str, predicciones))) + "\n")
                errores_predicciones_intervalos["Linear Regression"] += y_test_sum
                
                #MLP
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                mlp = MLPRegressor(hidden_layer_sizes=(13),max_iter=10000)
                mlp.fit(X_train,y_train)
                y_pred = mlp.predict(X_test)
                predicciones=[]
                y_test_sum = 0
                for index, fila in travel_time_dataframe.iterrows():
                        valores_predichos[(pair[0],pair[1],fila['date'].day,interval.strftime("%H:%M"))].append(math.ceil(y_pred[fila['date'].day - 18]))
                        predicciones.append(y_pred[fila['date'].day - 18])
                        y_test_sum += abs((fila['volume'] - math.ceil(y_pred[fila['date'].day - 18])) / fila['volume'])
                y_test_sum /= len(travel_time_dataframe);
                the_file.write("Red Neuronal, "+  str(','.join(map(str, predicciones))) + "\n")
                print("Predicciones : ", predicciones)
                errores_predicciones_intervalos["MLP"] += y_test_sum
                
                #SVR
                model = svm.SVR(C=100000,epsilon=0.1, gamma=0.00008,
                    kernel='rbf', verbose=False)
                model.fit(X_train, y_train) 
                y_pred = model.predict(X_test)
                print("SVR : ", y_pred)
                predicciones= [None] * 7
                y_test_sum = 0
                for index, fila in travel_time_dataframe.iterrows():
                        valores_predichos[(pair[0],pair[1],fila['date'].day,interval.strftime("%H:%M"))].append(math.ceil(y_pred[fila['date'].day - 18]))
                        predicciones[fila['date'].day - 18] = (y_pred[fila['date'].day - 18])
                        y_test_sum += abs((fila['volume'] - math.ceil(y_pred[fila['date'].day - 18])) / fila['volume'])
                y_test_sum /= len(travel_time_dataframe);
                the_file.write("Máquina de Soporte Vectorial,: "+ str(','.join(map(str, predicciones))) + "\n")
                errores_predicciones_intervalos["SVR"] += y_test_sum
               
                #KNN
                model =KNeighborsRegressor(n_neighbors=10)
                model.fit(X_train, y_train) 
                y_pred = model.predict(X_test)
                predicciones=[]
                y_test_sum = 0
                for i in range(7):
                        predicciones.append
                for index, fila in travel_time_dataframe.iterrows():
                        valores_predichos[(pair[0],pair[1],fila['date'].day,interval.strftime("%H:%M"))].append(math.ceil(y_pred[fila['date'].day - 18]))
                        predicciones.append(y_pred[fila['date'].day - 18])
                        y_test_sum += abs((fila['volume'] - y_pred[fila['date'].day - 18]) / fila['volume'])
                y_test_sum /= len(travel_time_dataframe);
                errores_predicciones_intervalos["KNN"] += y_test_sum
                the_file.write("KNN, "+  str(','.join(map(str, predicciones))) + "\n")
                #LightGBM
                lgb_train = lgb.Dataset(X_train, y_train)
                params = {
                'application' : 'fair',
                'task': 'train',
                'num_threads': 8,
                'boosting_type': 'rf',
                'objective': 'regression',
                'metric': {'l2', 'auc'},
                'num_leaves': 50,
                'learning_rate': 0.001,
                'feature_fraction': 0.3,
                'bagging_fraction': 0.3,
                'bagging_freq': 1,
                'min_data':1,
                'verbose': 0
                }
                gbm = lgb.train(params, lgb_train)
                y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
                y_test_sum = 0
                predicciones=[]
                for index, fila in travel_time_dataframe.iterrows():
                        valores_predichos[(pair[0],pair[1],fila['date'].day,interval.strftime("%H:%M"))].append(math.ceil(y_pred[fila['date'].day - 18]))
                        predicciones.append(y_pred[fila['date'].day - 18])
                        y_test_sum += abs((fila['volume'] - y_pred[fila['date'].day - 18]) / fila['volume'])
                y_test_sum /= len(travel_time_dataframe);
                errores_predicciones_intervalos["LightGBM"] += y_test_sum
                the_file.write("LightGBM, "+ str(','.join(map(str, predicciones))) + "\n\n")
             for key, value in errores_predicciones_rutas.items():
                errores_predicciones_rutas[key] += errores_predicciones_intervalos[key]/len(time_intervals)
                errores_predicciones_intervalos[key] = 0  
        for key, value in errores_predicciones_rutas.items():
                errores_predicciones_rutas[key] = errores_predicciones_rutas[key]/len(pairs);
        print ("Error : ", errores_predicciones_rutas);  

days = [18,19,20,21,22,23,24]
datos_predicciones = []
for pair in pairs:
     for interval in time_intervals:
        for day in days:
             fila = []
             direccion = 'Entrada';
             if (pair[1] == 1):
              direccion = 'Salida'
             fila = fila + ([(pair[0], direccion), day, (interval.strftime("%H:%M"), (interval+datetime.timedelta(minutes=20)).strftime("%H:%M"))])
             if ((pair[0],pair[1],day,interval.strftime("%H:%M")) in valores_predichos):
                fila = fila + valores_predichos[(pair[0],pair[1],day,interval.strftime("%H:%M"))]
             else:
                fila = fila + (['-'] * 7)
             print("FILA : ", fila)
             datos_predicciones.append(fila)
                       


tabla_predicciones = pd.DataFrame(datos_predicciones, columns=['Par','Intervalo de tiempo','Día' , 'Valor real', 'XGBoost', 'Linear Regression', 'LightGBM', 'MLP', 'SVR', 'KNN'])
print("TABLA PREDICCIONES : ", tabla_predicciones)
tabla_predicciones.to_html("tabla_predicciones.html")    

