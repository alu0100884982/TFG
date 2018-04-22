# En esta aproximación se realiza la predicción de los intervalos de tiempo a predecir de forma directa; es decir,  model_fit.forecast(steps=6)[0] => predice los 6 intervalos de golpe.
# ¿Se debería hacer la predicción poco a poco? Es decir, en vez de predecir los intervalos de golpe, ¿ se debería predecir uno y luego añadir el valor predicho a la serie temporal, volver a entrenar la serie temporal y predecir el siguiente valor?

import psycopg2
import pandas as pd
from pandas import set_option
from numpy import loadtxt
from pandas import read_csv
from pandas import datetime
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import datetime


def creacionElementosDiccionario(intervalo1, intervalo2, hora_del_dia, hora_de_referencia,day,df1_aux,route):
        try:
           conn = psycopg2.connect("dbname='tfgtest1' user='javisunami' host='localhost' password='javier123'")
        except:
           print("I am unable to connect to the database")
        cur = conn.cursor()
        query = "select time_window[1], avg_travel_time from travel_time_intersection_to_tollgate_test1 where intersection_id = '"+ str(route[0]) +"' AND tollgate_id = " + str(route[1]) + " AND (time_window[1].time BETWEEN " + intervalo1 + ") AND (time_window[1].date = DATE '"+str(day)+"') order by time_window;"
        cur.execute(query)
        rows = cur.fetchall()
        df2 = pd.DataFrame.from_records(rows, columns=['date','avg_travel_time'])
        result_dataframe = pd.concat([df1_aux,df2])
        try:
           conn = psycopg2.connect("dbname='tfgtraining2' user='javisunami' host='localhost' password='javier123'")
        except:
           print("I am unable to connect to the database")
        query = "select time_window[1], avg_travel_time from travel_time_intersection_to_tollgate_training2 where intersection_id = '"+ str(route[0]) +"' AND tollgate_id = " + str(route[1]) +  "AND (time_window[1].date = DATE '"+str(day)+"') AND (time_window[1].time BETWEEN " + intervalo2 + ") order by time_window;"
        cur = conn.cursor()
        cur.execute(query)
        rows2 = cur.fetchall()
        serie = pd.Series(result_dataframe['avg_travel_time'].values, index=result_dataframe['date'])
        best_score, best_cfg = float("inf"), None
        valores_reales = [element[1] for element in rows2]
        '''
        for p in range(3,10):
          for d in range(3):
            for q in range(5):   
                 print("ORDEN : ", (p,d,q))
                 orderr = (p,d,q)
                 try:    
                         model = ARIMA(serie, order=orderr)    
                         model_fit = model.fit(disp=0)
                         forecast = model_fit.forecast(steps=6)[0]
                         new_forecast=[]
                         for element in rows2:
                                  new_forecast.append(forecast[((datetime.datetime(2018,1,1,element[0].hour,element[0].minute, 0)-hora_de_referencia)/1200).seconds])
                         mse = mean_squared_error(valores_reales, new_forecast)
                         print("MSE : ", mse, " BEST_SCORE: ", best_score)
                         if mse < best_score:
                                print("BEST_SCORE : ", best_score)
                                best_score, best_cfg = mse, orderr
                 except:
                         continue
        print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score)) 
        '''
        model = ARIMA(serie, order= (9,1,0))
        model_fit = model.fit(disp=0)
        predicciones_ruta_dia[route[0], route[1],day,hora_del_dia] = model_fit.forecast(steps=6)[0]


try:
   conn = psycopg2.connect("dbname='tfgtraining2' user='javisunami' host='localhost' password='javier123'")
except:
   print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""SELECT * FROM travel_time_intersection_to_tollgate_training2 WHERE (time_window[1].time BETWEEN TIME '08:00:00' AND TIME '09:40:00') OR (time_window[1].time BETWEEN TIME '17:00:00' AND TIME '18:40:00') ORDER BY intersection_id, tollgate_id, time_window """)
rows = cur.fetchall()
colnames = ['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time']
intervals_to_predict_real_avgtraveltime = pd.DataFrame(rows, columns=colnames)
routes = np.array(intervals_to_predict_real_avgtraveltime.iloc[:,0:2].values.tolist())
time_intervals = np.array(intervals_to_predict_real_avgtraveltime.iloc[:,2].values.tolist())
routes = np.unique(routes, axis=0);
days = np.unique([time_interval[0].strftime("%Y-%m-%d") for time_interval in time_intervals], axis=0)
aux = np.array([])
for time_interval in time_intervals:
        time_interval[0] = datetime.datetime(2016, 10,18, time_interval[0].hour, time_interval[0].minute)
        aux = np.append(aux,time_interval[0]);
time_intervals = sorted(set(aux))


predicciones_ruta_dia = dict()
count = 0;
for route in routes:
        try:
                conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
        except:
               print("I am unable to connect to the database")
        cur = conn.cursor()
        query = "select time_window[1], avg_travel_time from travel_time_intersection_to_tollgate_modified  where intersection_id = '" +route[0] +"' AND tollgate_id = " + route[1] + " order by time_window;"
        cur.execute(query)
        rows = cur.fetchall()
        df1 = pd.DataFrame.from_records(rows, columns=['date','avg_travel_time'])
        #df1 = df1[(df1.avg_travel_time > 50) & (df1.avg_travel_time < 150)]
        minimum_date = min(df1.date)
        maximum_date = max(df1.date)
        date_aux = minimum_date
        while (date_aux != maximum_date):
               if (not((date_aux == df1['date']).any())):
                 valores_avg_travel = []
                 for row in df1.values:
                        if (row[0].time() == date_aux.time()):
                                valores_avg_travel.append(row[1])
                 df1.loc[len(df1)] = [date_aux, np.mean(valores_avg_travel)]
               date_aux += datetime.timedelta(minutes=20)
        df1 = df1.sort_index()
        for day in days:
              df1_aux = df1.copy()
              count += 1;
              print("CUENTA : ", count)
              minimum_date = maximum_date
              maximum_date = datetime.datetime(2016,10,int(day[8:10]),6,00,0)
              date_aux = minimum_date
              while (date_aux != maximum_date):
               if (not((date_aux == df1_aux['date']).any())):
                 valores_avg_travel = []
                 for row in df1_aux.values:
                        if (row[0].time() == date_aux.time()):
                                valores_avg_travel.append(row[1])
               df1_aux.loc[len(df1_aux)] = [date_aux, np.mean(valores_avg_travel)]
               date_aux += datetime.timedelta(minutes=20)
              df1_aux = df1_aux.sort_index()
              creacionElementosDiccionario("TIME '6:00:00' AND TIME '7:40:00'","TIME '8:00:00' AND TIME '9:40:00'", 0, datetime.datetime(2018,1,1,8,0,0),day,df1_aux,route)
              creacionElementosDiccionario("TIME '15:00:00' AND TIME '16:40:00'","TIME '17:00:00' AND TIME '18:40:00'", 1, datetime.datetime(2018,1,1,17,0,0),day,df1_aux,route)
              
for key,val in predicciones_ruta_dia.items():
    print (key, "=>", val)                
routes_sum = 0;   

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
                query = "select time_window[1].date, avg_travel_time from travel_time_intersection_to_tollgate_training2 where intersection_id = '"+ str(route[0]) +"' AND tollgate_id = " + str(route[1]) + " AND (time_window[1].time = TIME '" + interval.strftime("%H:%M:%S") + "') AND (time_window[1].date = DATE '"+str(day)+"') order by time_window;"
                cur = conn.cursor()
                cur.execute(query)
                rows2 = cur.fetchall()
                if (len(rows2) > 0):
                        lhs = datetime.datetime(2018,1,1,interval.hour,interval.minute,0)
                        momento_del_dia = 0;
                        if (interval.hour == 8 or interval.hour == 9):
                           rhs = datetime.datetime(2018,1,1,8,0,0)
                        else:
                           momento_del_dia = 1
                           rhs = datetime.datetime(2018,1,1,17,0,0)
                           print("FORECAST : ", predicciones_ruta_dia[route[0], route[1], day,momento_del_dia][((lhs-rhs)/1200).seconds])
                           print("ROWS2 : ",rows2[0][1])
                        y_test_sum += abs((rows2[0][1] - predicciones_ruta_dia[route[0], route[1], day,momento_del_dia][((lhs-rhs)/1200).seconds]) / rows2[0][1])
                        count += 1
           intervals_sum += y_test_sum/count;     
        routes_sum += intervals_sum /len(time_intervals)
print("Error MAPE : ", (routes_sum/len(routes)))
