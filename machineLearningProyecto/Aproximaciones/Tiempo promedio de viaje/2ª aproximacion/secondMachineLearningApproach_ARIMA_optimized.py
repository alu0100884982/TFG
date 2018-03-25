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
        df1 = df1[(df1.avg_travel_time > 50) & (df1.avg_travel_time < 150)]
        for day in days:
                try:
                   conn = psycopg2.connect("dbname='tfgtest1' user='javisunami' host='localhost' password='javier123'")
                except:
                   print("I am unable to connect to the database")
                cur = conn.cursor()
                query = "select time_window[1], avg_travel_time from travel_time_intersection_to_tollgate_test1 where intersection_id = '"+ str(route[0]) +"' AND tollgate_id = " + str(route[1]) + " AND (time_window[1].time BETWEEN TIME '6:00:00' AND TIME '7:40:00') AND (time_window[1].date = DATE '"+str(day)+"') order by time_window;"
                cur.execute(query)
                rows = cur.fetchall()
                df2 = pd.DataFrame.from_records(rows, columns=['date','avg_travel_time'])
                result_dataframe = pd.concat([df1,df2])
                serie = pd.Series(result_dataframe['avg_travel_time'].values, index=result_dataframe['date'])
                model = ARIMA(serie, order=(9,1,2))
                model_fit = model.fit(disp=0)
                predicciones_ruta_dia[route[0], route[1],day] = model_fit.forecast(steps=6)[0]
                
routes_sum = 0;    
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
        df1 = df1[(df1.avg_travel_time > 50) & (df1.avg_travel_time < 150)]
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
                        if (interval.hour == 8 or interval.hour == 9):
                           rhs = datetime.datetime(2018,1,1,8,0,0)
                        else:
                           rhs = datetime.datetime(2018,1,1,17,0,0)
                        y_test_sum += abs((rows2[0][1] - predicciones_ruta_dia[route[0], route[1], day][((lhs-rhs)/1200).seconds]) / rows2[0][1])
                        count += 1
           intervals_sum += y_test_sum/count;     
        routes_sum += intervals_sum /len(time_intervals)
print("Error MAPE : ", (routes_sum/len(routes)))
                
'''
for route in routes:
        routes_sum = 0;
        try:
                conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
        except:
               print("I am unable to connect to the database")
        cur = conn.cursor()
        query = "select time_window[1], avg_travel_time from travel_time_intersection_to_tollgate_modified  where intersection_id = '" +route[0] +"' AND tollgate_id = " + route[1] + " order by time_window;"
        cur.execute(query)
        rows = cur.fetchall()
        df1 = pd.DataFrame.from_records(rows, columns=['date','avg_travel_time'])
        df1 = df1[(df1.avg_travel_time > 50) & (df1.avg_travel_time < 150)]
        suma_intervalos_tiempo = 0;
        intervals_sum = 0;
        for interval in time_intervals:
           count = 0;
           y_test_sum = 0
           for day in days:
                print("DAY : ", day)
                try:
                   conn = psycopg2.connect("dbname='tfgtest1' user='javisunami' host='localhost' password='javier123'")
                except:
                   print("I am unable to connect to the database")
                cur = conn.cursor()
                query = "select time_window[1], avg_travel_time from travel_time_intersection_to_tollgate_test1 where intersection_id = '"+ str(route[0]) +"' AND tollgate_id = " + str(route[1]) + " AND (time_window[1].time BETWEEN TIME '6:00:00' AND TIME '7:40:00') AND (time_window[1].date = DATE '"+str(day)+"') order by time_window;"
                cur.execute(query)
                rows = cur.fetchall()
                df2 = pd.DataFrame.from_records(rows, columns=['date','avg_travel_time'])
                result_dataframe = pd.concat([df1,df2])

                try:
                   conn = psycopg2.connect("dbname='tfgtraining2' user='javisunami' host='localhost' password='javier123'")
                except:
                   print("I am unable to connect to the database")
                query = "select time_window[1].date, avg_travel_time from travel_time_intersection_to_tollgate_training2 where intersection_id = '"+ str(route[0]) +"' AND tollgate_id = " + str(route[1]) + " AND (time_window[1].time = TIME '" + interval.strftime("%H:%M:%S") + "') AND (time_window[1].date = DATE '"+str(day)+"') order by time_window;"
                cur = conn.cursor()
                cur.execute(query)
                rows2 = cur.fetchall()
                if (len(rows2) > 0):
                        serie = pd.Series(result_dataframe['avg_travel_time'].values, index=result_dataframe['date'])
                        best_score, best_cfg = float("inf"), None
                        for p in range(3,10):
                          for d in range(3):
                            for q in range(5):   
                                 print("ORDEN : ", (p,d,q))
                                 orderr = (p,d,q)
                                 model = ARIMA(serie, order=orderr)
                                 try: 
                                         model_fit = model.fit(disp=0)
                                         forecast = model_fit.forecast(steps=6)[0]
                                         error = mean_squared_error(rows2, forecast)
                                         print("ROWS2: ", rows2)
                                         print("FORECAST: ", forecast)
                                         mse = mean_squared_error(rows2, forecast)
                                         if mse < best_score:
                                                best_score, best_cfg = mse, orderr
                                         print("ERROR: ", error)
                                         
                                 except:
                                         continue
                                         
                        print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score)) 
                        model = ARIMA(serie, order=(9,1,2))
                        model_fit = model.fit(disp=0)
                        forecast = model_fit.forecast(steps=6)[0]
                        lhs = datetime.datetime(2018,1,1,interval.hour,interval.minute,0)
                        if (interval.hour == 8 or interval.hour == 9):
                          rhs = datetime.datetime(2018,1,1,8,0,0)
                        else:
                          rhs = datetime.datetime(2018,1,1,17,0,0)
                        print("FORECAST :", forecast[((lhs-rhs)/1200).seconds])
                        print("REAL: ", rows2[0][1])
                        print("DIFFERENCE : ", abs((rows2[0][1] - forecast[((lhs-rhs)/1200).seconds]) / rows2[0][1]))      
                        y_test_sum += abs((rows2[0][1] - forecast[((lhs-rhs)/1200).seconds]) / rows2[0][1])
                        count += 1
           intervals_sum += y_test_sum/count;
           print("intervals_sum : ",  intervals_sum, " count: ", count )    
        routes_sum += intervals_sum /len(time_intervals)
        print("routes_sum : ",  routes_sum, " lenth_intervals: ", len(time_intevals))  
print("Error MAPE : ", (routes_sum/len(routes)))
'''      


     
     
     
