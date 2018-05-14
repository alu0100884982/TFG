import psycopg2
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import datetime
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
'''
# Prueba de MLP con las columnas date-avg_travel_time de entrenamiento
try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")
cur = conn.cursor()
query = "select time_window[1], avg_travel_time from travel_time_intersection_to_tollgate_modified  where intersection_id = 'A' AND tollgate_id = 2 order by time_window;"
cur.execute(query)
rows = cur.fetchall()
dates_traveltime = pd.DataFrame.from_records(rows, columns=['date','avg_travel_time'])
minimum_date = min(dates_traveltime.date)
maximum_date = max(dates_traveltime.date)
date_aux = minimum_date
dates_traveltime = dates_traveltime[(dates_traveltime['avg_travel_time'] > 30) & (dates_traveltime['avg_travel_time'] < 150)]
dates_traveltime = dates_traveltime.reset_index(drop=True)
while (date_aux != maximum_date): 
       if (not((date_aux == dates_traveltime['date']).any())):
         valores_avg_travel = []
         for row in dates_traveltime.values:
                if (row[0].time() == date_aux.time()):
                        valores_avg_travel.append(row[1])
         dates_traveltime.loc[len(dates_traveltime)] = [date_aux, np.mean(valores_avg_travel)]
       date_aux += datetime.timedelta(minutes=20)
dates_traveltime = dates_traveltime.sort_index()
series = pd.Series(dates_traveltime['avg_travel_time'].values, index=dates_traveltime['date'])
dataframe = pd.DataFrame()
for i in range(6,0,-1):
        dataframe['t-'+str(i)] = series.shift(i)
dataframe['t'] = series.values
dataframe = dataframe[10:]
#print("HEAD : ", dataframe.head())
array = dataframe.values
X = array[:,0:6]
y = array[:,6]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=7)
mlp = MLPRegressor(hidden_layer_sizes=(5,18,18),max_iter=4000)
mlp.fit(X_train,y_train)
indexes = [i for i in range(len(X_test))]
predictions = mlp.predict(X_test)
error =0;
for i in range(len(predictions)):
 error += abs(y_test[i] - predictions[i])/y_test[i]
 print("PREDICTION: ", predictions[i], " REAL : ", y_test[i])
print("ERROR : ", (error/len(predictions)))
plt.plot(indexes,predictions, color='blue')
plt.plot(indexes,y_test, color='black')
plt.show()
'''

#################################################Prediction task#####################################

def xgboost(X_train, y_train):
     modelo = XGBRegressor()
     modelo.fit(X_train, y_train)
     return modelo, "XGBoost"
 
def lightgbm(X_train, y_train):
     lgb_train = lgb.Dataset(X_train, y_train)
     params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': {'l2', 'auc'},
                'num_leaves': 40,
                'num_iterations': 4000,
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0
              }
     lgbm = lgb.train(params, lgb_train)
     return lgbm, "LightGBM"
 
def linearregression(X_train, y_train):
     modelo = linear_model.LinearRegression()
     modelo.fit(X_train, y_train)
     return modelo, "Linear Regression"
 
def svr(X_train, y_train):
    modelo = SVR(C=1000,epsilon=0.3, gamma=0.008, kernel='rbf', verbose=False)
    modelo.fit(X_train, y_train) 
    return modelo, "SVR"
 
def neuralnetworks(X_train, y_train):
  # scaler = StandardScaler()
  # scaler.fit(X_train)
   #X_train = scaler.transform(X_train)
   modelo = MLPRegressor(hidden_layer_sizes=(24,24,24),max_iter=4000)
   modelo.fit(X_train,y_train)
   return modelo, "NN"

def knn(X_train, y_train):
   modelo =KNeighborsRegressor(n_neighbors=20)
   modelo.fit(X_train, y_train)
   return modelo, "KNN"

valores_predichos = {}
   
for j in range(6):
        routes = [ ('C', 1) ]
        days = list(range(24,25))
        intervals_2hours_previous = [(6,8),(15,17)]
        intervals_to_predict = ['08:00-10:00','17:00-19:00']
        number_intervals_to_predict = 6
        predictions = dict()

        for route in routes:
                try:
                      conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
                except:
                      print("I am unable to connect to the database")
                cur = conn.cursor()
                query = "select time_window[1], avg_travel_time from travel_time_intersection_to_tollgate_modified  where intersection_id = '" + str(route[0]) + "' AND tollgate_id = " + str(route[1]) + " order by time_window;"
                cur.execute(query)
                rows_training = cur.fetchall()
                dates_traveltime = pd.DataFrame.from_records(rows_training, columns=['date','avg_travel_time'])
                for day in days:
                  for interval in intervals_2hours_previous:
                          minimum_date = min(dates_traveltime.date)
                          maximum_date = datetime.datetime(2016,10,17,23,40,0)
                          dates_traveltime_original = pd.DataFrame(dates_traveltime)
                          dates_traveltime = dates_traveltime[(dates_traveltime['avg_travel_time'] > 30) & (dates_traveltime['avg_travel_time'] < 150)]
                          dates_traveltime = dates_traveltime.reset_index(drop=True) 
                          date_aux = minimum_date

                          
                          while (date_aux != maximum_date): 
                            if (not((date_aux == dates_traveltime['date']).any())):
                              valores_avg_travel = []
                              for row in  dates_traveltime_original.values:
                                if (row[0].time() == date_aux.time()):
                                        valores_avg_travel.append(row[1])
                              dates_traveltime.loc[len(dates_traveltime)] = [date_aux, np.mean(valores_avg_travel)]
                            date_aux += datetime.timedelta(minutes=20)
                          dates_traveltime = dates_traveltime.sort_index()
                          
                          for row in dates_traveltime.values:
                                print("ROW : ", row)
                          
                          if (interval[0] == 6):
                           minimum_date = datetime.datetime(2016,10,day,0,0,0)
                           maximum_date = datetime.datetime(2016,10,day,6,0,0)
                          else:
                           minimum_date = datetime.datetime(2016,10,day,0,0,0)
                           maximum_date = datetime.datetime(2016,10,day,15,0,0)
                          
                          date_aux = minimum_date
                          dates_traveltime_filled = pd.DataFrame(dates_traveltime)
      
                          
                          while (date_aux != maximum_date): 
                            if (not((date_aux == dates_traveltime_filled['date']).any())):
                              valores_avg_travel = []
                              for row in dates_traveltime_filled.values:
                                if (row[0].time() == date_aux.time() and row[0].weekday() == date_aux.weekday()):
                                        valores_avg_travel.append(row[1])
                              #print("VALORES : ", valores_avg_travel)
                              dates_traveltime_filled.loc[len(dates_traveltime_filled)] = [date_aux, np.mean(valores_avg_travel)]
                            date_aux += datetime.timedelta(minutes=20)
                          dates_traveltime_filled = dates_traveltime_filled.sort_index()
                          try:
                              conn = psycopg2.connect("dbname='tfgtest1' user='javisunami' host='localhost' password='javier123'")
                          except:
                              print("I am unable to connect to the database")
                          cur = conn.cursor()
                          query = "select time_window[1], avg_travel_time from travel_time_intersection_to_tollgate_test1  where intersection_id = '" + str(route[0]) + "' AND tollgate_id = " + str(route[1]) + " AND extract(day from time_window[1]) = " + str(day) + " AND extract(hour from time_window[1]) BETWEEN " + str(interval[0]) + " AND " + str((interval[1] - 1)) + " order by time_window;"
                          cur.execute(query)
                          row_2hoursintervals_before = cur.fetchall()
                          dates_traveltime_2hoursintervals_before = pd.DataFrame.from_records(row_2hoursintervals_before, columns=['date','avg_travel_time'])
                          dates_traveltime_filled = pd.concat([dates_traveltime_filled,dates_traveltime_2hoursintervals_before])
                          print("DATES_TRAVELTIME: ", dates_traveltime_filled)
                          series_dates_traveltime_filled = pd.Series(dates_traveltime_filled['avg_travel_time'].values, index=dates_traveltime_filled['date'])
                          dates_traveltime_supervised = pd.DataFrame()
                          number_time_steps_previous = 5
                          for i in range(number_time_steps_previous,0,-1):
                                dates_traveltime_supervised['t-'+str(i)] = series_dates_traveltime_filled.shift(i)
                          dates_traveltime_supervised['t'] = series_dates_traveltime_filled .values
                          dates_traveltime_supervised = dates_traveltime_supervised[number_time_steps_previous:]
                          print(dates_traveltime_supervised.head(20).to_string(index=False))
                          X_train = dates_traveltime_supervised.iloc[:,0:number_time_steps_previous]
                          y_train = dates_traveltime_supervised.iloc[:,number_time_steps_previous]
                          print("X_train : ", X_train)
                           #Elegimos el modelo
                          if (j == 0):
                                modelo, nombre_algoritmo = xgboost(X_train, y_train)
                          elif (j == 1):
                                modelo, nombre_algoritmo = lightgbm(X_train, y_train) 
                          elif (j == 2):
                                modelo, nombre_algoritmo = linearregression(X_train, y_train)
                          elif (j == 3):
                                modelo, nombre_algoritmo = svr(X_train, y_train)
                          elif (j == 4):
                                modelo, nombre_algoritmo = knn(X_train, y_train)
                          elif (j == 5):
                                modelo, nombre_algoritmo = neuralnetworks(X_train, y_train)
                          
                          previous_row_prediction = dates_traveltime_supervised.iloc[-1].shift(-1).values[0:-1]
                          for k in range(number_intervals_to_predict):
                              dataframe_input = pd.DataFrame(previous_row_prediction).T
                              dataframe_input.columns = ['t-5','t-4','t-3','t-2', 't-1']
                             # print("INPUT : ", ((route[0],route[1],day,intervals_to_predict[0]) not in predictions.keys()))
                              original = 0   
                              #if(j == 5):
                                   # original = modelo.predict(dataframe_input)
                                  #  prediction = scaler.inverse_transform(original) 
                              #else:
                              prediction = modelo.predict(dataframe_input)
                              if (interval[0] == 6):
                               if((route[0],route[1],day,intervals_to_predict[0]) not in predictions.keys()):
                                  predictions[route[0],route[1],day,intervals_to_predict[0]] = prediction;
                               else:
                                   predictions[route[0],route[1],day,intervals_to_predict[0]] = np.append(predictions[(route[0],route[1],day,intervals_to_predict[0])], prediction);
                              else:
                                if((route[0],route[1],day,intervals_to_predict[1]) not in predictions.keys()):
                                  predictions[(route[0],route[1],day,intervals_to_predict[1])] = prediction;
                                else:
                                  predictions[(route[0],route[1],day,intervals_to_predict[1])] = np.append(predictions[(route[0],route[1],day,intervals_to_predict[1])], prediction);
                            #  if(j == 5):
                             #   prediction = original
                              previous_row_prediction = pd.DataFrame(np.append(previous_row_prediction, prediction)).shift(-1).values[0:-1]
                                   
                          #for key,val in predictions.items():
                             #    print(key, "=>", val)

        try:
           conn = psycopg2.connect("dbname='tfgtraining2' user='javisunami' host='localhost' password='javier123'")
        except:
           print("I am unable to connect to the database")

        cur = conn.cursor()
        cur.execute("""SELECT * FROM travel_time_intersection_to_tollgate_training2 WHERE (time_window[1].time BETWEEN TIME '08:00:00' AND TIME '09:40:00') OR (time_window[1].time BETWEEN TIME '17:00:00' AND TIME '18:40:00') ORDER BY intersection_id, tollgate_id, time_window """)
        rows = cur.fetchall()
        colnames = ['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time']
        dataframe = pd.DataFrame(rows, columns=colnames)
        time_intervals = np.array(dataframe.iloc[:,2].values.tolist())
        aux = np.array([])
        for time_interval in time_intervals:
                time_interval[0] = datetime.datetime(2016, 10,18, time_interval[0].hour, time_interval[0].minute)
                aux = np.append(aux,time_interval[0]);
        time_intervals = sorted(set(aux))

                 
        routes_sum = 0;   
        datos_predicciones = []

        for route in routes:
                suma_intervalos_tiempo = 0;
                intervals_sum = 0;
                for interval in time_intervals:
                   count = 0;
                   y_test_sum = 0
                   for day in days:
                        try:
                           conn = psycopg2.connect("dbname='tfgtraining2' user='javisunami' host='localhost' password='javier123'")
                        except:
                           print("I am unable to connect to the database")
                        query = "select time_window[1].date, avg_travel_time from travel_time_intersection_to_tollgate_training2 where intersection_id = '"+ str(route[0]) +"' AND tollgate_id = " + str(route[1]) + " AND (time_window[1].time = TIME '" + interval.strftime("%H:%M:%S") + "') AND (time_window[1].date = DATE '2016-10-"+str(day)+"') order by time_window;"
                        cur = conn.cursor()
                        cur.execute(query)
                        rows2 = cur.fetchall()
                        
                        if (len(rows2) > 0):
                                print("HOLA")
                                lhs = datetime.datetime(2018,1,1,interval.hour,interval.minute,0)
                                momento_del_dia = intervals_to_predict[0];
                                if (interval.hour == 8 or interval.hour == 9):
                                   rhs = datetime.datetime(2018,1,1,8,0,0)
                                else:
                                   momento_del_dia = intervals_to_predict[1]
                                   rhs = datetime.datetime(2018,1,1,17,0,0)
                                   print("FORECAST : ", predictions[route[0], route[1], day,momento_del_dia][((lhs-rhs)/1200).seconds])
                                   print("ROWS2 : ",rows2[0][1])
                                   
                                if (j == 0):
                                     valores_predichos[(route[0], route[1], day,interval.strftime("%H:%M"), (interval + datetime.timedelta(minutes=20)).strftime("%H:%M"))] = [rows2[0][1], float(predictions[route[0], route[1], day,momento_del_dia][((lhs-rhs)/1200).seconds])]
                                else:
                                      valores_predichos[(route[0], route[1], day,interval.strftime("%H:%M"), (interval + datetime.timedelta(minutes=20)).strftime("%H:%M"))] += [ float(predictions[route[0], route[1], day,momento_del_dia][((lhs-rhs)/1200).seconds])]
                                      
                                y_test_sum += abs((rows2[0][1] - predictions[route[0], route[1], day,momento_del_dia][((lhs-rhs)/1200).seconds]) / rows2[0][1])
                                count += 1
                                print("COUNT : ", count)

                        elif ((route[0], route[1], day,interval.strftime("%H:%M"), (interval + datetime.timedelta(minutes=20)).strftime("%H:%M")) not in valores_predichos):
                                valores_predichos[(route[0], route[1], day,interval.strftime("%H:%M"), (interval + datetime.timedelta(minutes=20)).strftime("%H:%M"))] = (['-'] * 7)
                   intervals_sum += y_test_sum/count;     
                routes_sum += intervals_sum /len(time_intervals)
        print("Error MAPE ", nombre_algoritmo, " :" , (routes_sum/len(routes)))
        

datos_predicciones = []
for route in routes:
     for interval in time_intervals:
        for day in days:
             fila = []
             fila += ([(route[0], route[1]), day, (interval.strftime("%H:%M"), (interval+datetime.timedelta(minutes=20)).strftime("%H:%M"))])
             if ((route[0], route[1], day,interval.strftime("%H:%M"), (interval + datetime.timedelta(minutes=20)).strftime("%H:%M")) in valores_predichos):
                 fila += valores_predichos[(route[0], route[1], day,interval.strftime("%H:%M"), (interval + datetime.timedelta(minutes=20)).strftime("%H:%M"))]
             datos_predicciones.append(fila)
               
        
tabla_predicciones = pd.DataFrame(datos_predicciones, columns=['Ruta','Día', 'Intervalo de tiempo' , 'Valor real', 'XGBoost', 'LightGBM', 'Linear Regression', 'SVR', 'KNN', 'MLP'])
print("TABLA PREDICCIONES : ", tabla_predicciones)
tabla_predicciones.to_html("tabla_predicciones.html")





















