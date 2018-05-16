import psycopg2
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model

#Función que crea una columna de booleanos para eliminar aquellas filas en las que se encuentren fechas con ruido.  
def crearBooleanos(input_array):
    booleanos = [] 
    for element in  dates_trafficvolume.date:
          if(str(element.date()) not in input_array):
            booleanos.append(True)
          else:
            booleanos.append(False)
    return pd.Series(booleanos)

#Función que diferencia una serie temporal para hacerla estacionaria.
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset['traffic_volume'].values)):
		value = dataset['traffic_volume'].values[i] - dataset['traffic_volume'].values[i - interval]
		diff.append(value)
	return pd.Series(diff)


# invert differenced forecast
def inverse_difference(last_ob, value):
	return value + last_ob

# Funciones que entrenan los distintos algoritmos de minería de datos.
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
    modelo = SVR(C=100000,epsilon=0.3, gamma=0.00008, kernel='rbf', verbose=False)
    modelo.fit(X_train, y_train) 
    return modelo, "SVR"
 
def neuralnetworks(X_train, y_train):
   #scaler = StandardScaler()
   #scaler.fit(X_train)
  # X_train = scaler.transform(X_train)
   modelo = MLPRegressor(hidden_layer_sizes=(24,24,24),max_iter=4000)
   modelo.fit(X_train,y_train)
   return modelo, "NN"

def knn(X_train, y_train):
   modelo =KNeighborsRegressor(n_neighbors=20)
   modelo.fit(X_train, y_train)
   return modelo, "KNN"




####Predicción del volumen de tráfico: de series temporales a aprendizaje supervisado.

valores_predichos = {}
for j in range(6):
        pairs = [(1,0),(1,1), (2,0),(3,0), (3,1)]
        days = list(range(18,25))
        intervals_2hours_previous = [(6,8),(15,17)]
        intervals_to_predict = ['08:00-10:00','17:00-19:00']
        number_intervals_to_predict = 6
        predictions = dict()
        #Fechas a quitar de los datos debido a que son ruido.
        dates_out_1_0 = ['2016-09-21','2016-09-28','2016-09-30','2016-10-01', '2016-10-02','2016-10-03','2016-10-04','2016-10-05','2016-10-06', '2016-10-07']
        dates_out_1_1 = ['2016-10-01', '2016-10-02','2016-10-03','2016-10-04','2016-10-05','2016-10-06', '2016-10-07']
        dates_out_2_0 = ['2016-09-28', '2016-10-01', '2016-10-02','2016-10-03','2016-10-04','2016-10-05','2016-10-06', '2016-10-07']

        for pair in pairs:
                try:
                      conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
                except:
                      print("I am unable to connect to the database")
                cur = conn.cursor()
                query = "select time_window[1], volume from traffic_volume_tollgates_modified  where tollgate_id = " + str(pair[0]) + " AND direction= " + str(pair[1]) + " order by time_window;"
                cur.execute(query)
                rows_training = cur.fetchall()
                dates_trafficvolume = pd.DataFrame.from_records(rows_training, columns=['date','traffic_volume'])
                dates_trafficvolume_original = pd.DataFrame.from_records(rows_training, columns=['date','traffic_volume'])
                if (pair[0] == 1 and pair[1] == 0):
                 booleanos = crearBooleanos(dates_out_1_0)
                 dates_trafficvolume = dates_trafficvolume[booleanos]
                elif ((pair[0] == 1 and pair[1] == 1) or (pair[0] == 3 and pair[1] == 1)):
                 booleanos = crearBooleanos(dates_out_1_1)
                 dates_trafficvolume = dates_trafficvolume[booleanos]
                elif (pair[0] == 2 and pair[1] == 0):
                 booleanos = crearBooleanos(dates_out_2_0)
                 dates_trafficvolume = dates_trafficvolume[booleanos]
                 
                dates_trafficvolume = dates_trafficvolume.reset_index(drop=True)

              
                #Gráfica de autocorrelación de la serie temporal
                '''
                plot_acf(series_decomposition)
                plot_pacf(series_decomposition)
                plt.show()
                '''
                #Descomposición de la serie temporal en sus tres componentes
                '''
                decomposition = seasonal_decompose(series_decomposition,freq=72)
                fig = decomposition.plot()  
                plt.show()
                plt.close()
                '''

                minimum_date = min(dates_trafficvolume.date)
                maximum_date = datetime.datetime(2016,10,17,0,0,0)
                date_aux = minimum_date
          
                while (date_aux != maximum_date): 
                    if (not((date_aux == dates_trafficvolume['date']).any())):
                      valores_volume = []
                      for row in dates_trafficvolume_original.values:
                        if (row[0].time() == date_aux.time()):
                                valores_volume.append(row[1])
                     # print(" PAIR : ", pair , " ", date_aux, "-> VALORES : ", valores_volume)
                      dates_trafficvolume.loc[len(dates_trafficvolume)] = [date_aux, np.mean(valores_volume)]
                    date_aux += datetime.timedelta(minutes=20)
                dates_trafficvolume = dates_trafficvolume.sort_values(by='date')

                  
                for day in days:
                  for interval in intervals_2hours_previous:
                                
                          if (interval[0] == 6):
                           minimum_date = datetime.datetime(2016,10,day,0,0,0)
                           maximum_date = datetime.datetime(2016,10,day,6,0,0)
                          else:
                           minimum_date = datetime.datetime(2016,10,day,0,0,0)
                           maximum_date = datetime.datetime(2016,10,day,15,0,0)
                           
                          date_aux = minimum_date
                          dates_trafficvolume_filled = pd.DataFrame(dates_trafficvolume, index = dates_trafficvolume.index)
                          
                          while (date_aux != maximum_date): 
                            if (not((date_aux == dates_trafficvolume_filled['date']).any())):
                              valores_avg_travel = []
                              for row in dates_trafficvolume_original.values:
                                if (row[0].time() == date_aux.time() ):
                                        valores_avg_travel.append(row[1])
                              dates_trafficvolume_filled.loc[len(dates_trafficvolume_filled)] = [date_aux, np.mean(valores_avg_travel)]
                            date_aux += datetime.timedelta(minutes=20)

                          dates_trafficvolume_filled = dates_trafficvolume_filled.sort_values(by='date')
                          
                          try:
                              conn = psycopg2.connect("dbname='tfgtest1' user='javisunami' host='localhost' password='javier123'")
                          except:
                              print("I am unable to connect to the database")
                          cur = conn.cursor()
                          query = "select time_window[1], volume from traffic_volume_tollgates_test1  where tollgate_id = " + str(pair[0]) + " AND direction = " + str(pair[1]) + " AND extract(day from time_window[1]) = " + str(day) + " AND extract(hour from time_window[1]) BETWEEN " + str(interval[0]) + " AND " + str((interval[1] - 1)) + " order by time_window;"
                          cur.execute(query)
                          row_2hoursintervals_before = cur.fetchall()
                          dates_trafficvolume_2hoursintervals_before = pd.DataFrame.from_records(row_2hoursintervals_before, columns=['date','traffic_volume'])
                          dates_trafficvolume_filled = pd.concat([dates_trafficvolume_filled,dates_trafficvolume_2hoursintervals_before])
                          
                          series_dates_trafficvolume_filled = pd.Series(dates_trafficvolume_filled['traffic_volume'].values, index=dates_trafficvolume_filled['date'])
                          dates_trafficvolume_supervised = pd.DataFrame()
                          number_time_steps_previous = 5
                          for i in range(number_time_steps_previous,0,-1):
                                dates_trafficvolume_supervised['t-'+str(i)] = series_dates_trafficvolume_filled.shift(i)
                          dates_trafficvolume_supervised['t'] = series_dates_trafficvolume_filled.values
                          

                          dates_trafficvolume_supervised = dates_trafficvolume_supervised[number_time_steps_previous:]
                          X_train = dates_trafficvolume_supervised.iloc[:,0:number_time_steps_previous]
                          y_train = dates_trafficvolume_supervised.iloc[:,number_time_steps_previous]
 

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

            
                          
                          #Guardamos los valores de la serie un día antes para poder volver a obtener el valor real de las predicciones realizadas.
                          #real_values = dates_trafficvolume_filled['traffic_volume'].values[-72:-66]
                          #series_dates_trafficvolume_filled = difference(dates_trafficvolume_filled,72)
                          
                          previous_row_prediction = dates_trafficvolume_supervised.iloc[-1].shift(-1).values[0:-1]
                          previous_row_prediction = [element for element in previous_row_prediction]
                          for k in range(number_intervals_to_predict):
                              dataframe_input = pd.DataFrame(previous_row_prediction).T
                              dataframe_input.columns = ['t-5','t-4','t-3','t-2', 't-1']
                              prediction = round(modelo.predict(dataframe_input)[0])
                              #prediction = inverse_difference(real_values[j],prediction)
                              if (interval[0] == 6):
                               if((pair[0],pair[1],day,intervals_to_predict[0]) not in predictions.keys()):
                                  predictions[pair[0],pair[1],day,intervals_to_predict[0]] = prediction;
                               else:
                                   predictions[pair[0],pair[1],day,intervals_to_predict[0]] = np.append(predictions[(pair[0],pair[1],day,intervals_to_predict[0])], prediction);
                              else:
                                if((pair[0],pair[1],day,intervals_to_predict[1]) not in predictions.keys()):
                                  predictions[(pair[0],pair[1],day,intervals_to_predict[1])] = prediction;
                                else:
                                  predictions[(pair[0],pair[1],day,intervals_to_predict[1])] = np.append(predictions[(pair[0],pair[1],day,intervals_to_predict[1])], prediction);
                              previous_row_prediction = pd.DataFrame(np.append(previous_row_prediction, prediction)).shift(-1).values[0:-1]       
                         # for key,val in predictions.items():
                                # print(key, "=>", val)
                         # print("\n")



        #Obtención de los intervalos a predecir
        try:
           conn = psycopg2.connect("dbname='tfgtraining2' user='javisunami' host='localhost' password='javier123'")
        except:
           print("I am unable to connect to the database")

        cur = conn.cursor()
        cur.execute("""SELECT time_window FROM traffic_volume_tollgates_training2 WHERE (time_window[1].time BETWEEN TIME '08:00:00' AND TIME '09:40:00') OR (time_window[1].time BETWEEN TIME '17:00:00' AND TIME '18:40:00') ORDER BY time_window """)
        rows = cur.fetchall()
        colnames = ['time_window']
        dataframe = pd.DataFrame(rows, columns=colnames)
        time_intervals = np.array(dataframe.iloc[:,0].values.tolist())
        aux = np.array([])
        for time_interval in time_intervals:
                time_interval[0] = datetime.datetime(2016, 10,18, time_interval[0].hour, time_interval[0].minute)
                aux = np.append(aux,time_interval[0]);
        time_intervals = sorted(set(aux))


        #Cálculo del error de las predicciones
                 
        pairs_sum = 0  
        forecasts = []
        real_values = []
        for pair in pairs:
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
                        query = "select time_window[1].date, volume from traffic_volume_tollgates_training2 where tollgate_id = "+ str(pair[0]) +" AND direction = " + str(pair[1]) + " AND (time_window[1].time = TIME '" + interval.strftime("%H:%M:%S") + "') AND (time_window[1].date = DATE '2016-10-"+str(day)+"') order by time_window;"
                        cur = conn.cursor()
                        cur.execute(query)
                        rows2 = cur.fetchall()
                        if (len(rows2) > 0):
                                lhs = datetime.datetime(2018,1,1,interval.hour,interval.minute,0)
                                momento_del_dia = intervals_to_predict[0];
                                if (interval.hour == 8 or interval.hour == 9):
                                   rhs = datetime.datetime(2018,1,1,8,0,0)
                                else:
                                   momento_del_dia = intervals_to_predict[1]
                                   rhs = datetime.datetime(2018,1,1,17,0,0)
                                   #print("FORECAST : ", predictions[pair[0], pair[1], day,momento_del_dia][((lhs-rhs)/1200).seconds])
                                   #print("ROWS2 : ",rows2[0][1])
                                if (j == 0):
                                 valores_predichos[(pair[0], pair[1], day,interval.strftime("%H:%M"), (interval + datetime.timedelta(minutes=20)).strftime("%H:%M"))] = [rows2[0][1], round(float(predictions[pair[0], pair[1], day,momento_del_dia][((lhs-rhs)/1200).seconds]))]
                                else:
                                 valores_predichos[(pair[0], pair[1], day,interval.strftime("%H:%M"), (interval + datetime.timedelta(minutes=20)).strftime("%H:%M"))] += [ round(float(predictions[pair[0], pair[1], day,momento_del_dia][((lhs-rhs)/1200).seconds]))]
                                forecasts.append(round(predictions[pair[0], pair[1], day,momento_del_dia][((lhs-rhs)/1200).seconds]))
                                real_values.append(rows2[0][1])
                                y_test_sum += abs((rows2[0][1] - predictions[pair[0], pair[1], day,momento_del_dia][((lhs-rhs)/1200).seconds]) / rows2[0][1])
                                count += 1
                   intervals_sum += y_test_sum/count;     
                pairs_sum += intervals_sum /len(time_intervals)
        print("Error MAPE ", nombre_algoritmo, " :", (pairs_sum/len(pairs)))

        #indexes = [i for i in range(forecasts)]
        #plt.plot(indexes,forecasts, color='blue')
        #plt.plot(indexes,real_values, color='black')
        #plt.show()
        
datos_predicciones = []
for pair in pairs:
     for interval in time_intervals:
        for day in days:
             fila = []
             fila += ([(pair[0], pair[1]), day, (interval.strftime("%H:%M"), (interval+datetime.timedelta(minutes=20)).strftime("%H:%M"))])
             if ((pair[0], pair[1], day,interval.strftime("%H:%M"), (interval + datetime.timedelta(minutes=20)).strftime("%H:%M")) in valores_predichos):
                 fila += valores_predichos[(pair[0], pair[1], day,interval.strftime("%H:%M"), (interval + datetime.timedelta(minutes=20)).strftime("%H:%M"))]
             datos_predicciones.append(fila)
               
        
tabla_predicciones = pd.DataFrame(datos_predicciones, columns=['Ruta','Día', 'Intervalo de tiempo' , 'Valor real', 'XGBoost', 'LightGBM', 'Linear Regression', 'SVR', 'KNN', 'MLP'])
#print("TABLA PREDICCIONES : ", tabla_predicciones)
tabla_predicciones.to_html("tabla_predicciones.html")

