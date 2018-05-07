import psycopg2
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import datetime
import matplotlib.pyplot as plt

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

routes = [('A',2),('A',3)]
days = list(range(18,25))
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
                  if (interval[0] == 6):
                   maximum_date = datetime.datetime(2016,10,day,6,0,0)
                  else:
                   maximum_date = datetime.datetime(2016,10,day,15,0,0)
                  date_aux = minimum_date
                  dates_traveltime = dates_traveltime[(dates_traveltime['avg_travel_time'] > 30) & (dates_traveltime['avg_travel_time'] < 150)]
                  dates_traveltime = dates_traveltime.reset_index(drop=True)
                  dates_traveltime_filled = pd.DataFrame(dates_traveltime)
                  while (date_aux != maximum_date): 
                    if (not((date_aux == dates_traveltime_filled['date']).any())):
                      valores_avg_travel = []
                      for row in dates_traveltime_filled.values:
                        if (row[0].time() == date_aux.time() and row[0].weekday() == date_aux.weekday()):
                                valores_avg_travel.append(row[1])
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
                  #print("DATES_TRAVELTIME: ", dates_traveltime_filled)
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
                  mlp = MLPRegressor(hidden_layer_sizes=(24,24,24),max_iter=4000)
                  mlp.fit(X_train,y_train)
                  
                  previous_row_prediction = dates_traveltime_supervised.iloc[-1].shift(-1).values[0:-1]
                  for j in range(number_intervals_to_predict):
                      print("J: ", j)
                      dataframe_input = pd.DataFrame(previous_row_prediction).T
                     # print("INPUT : ", ((route[0],route[1],day,intervals_to_predict[0]) not in predictions.keys()))
                      prediction = mlp.predict(dataframe_input)
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
                      previous_row_prediction = pd.DataFrame(np.append(previous_row_prediction, prediction)).shift(-1).values[0:-1]
                           
                  for key,val in predictions.items():
                         print(key, "=>", val)



#Obtención de los intervalos a predecir
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

#Cálculo del error de las predicciones
         
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
                query = "select time_window[1].date, avg_travel_time from travel_time_intersection_to_tollgate_training2 where intersection_id = '"+ str(route[0]) +"' AND tollgate_id = " + str(route[1]) + " AND (time_window[1].time = TIME '" + interval.strftime("%H:%M:%S") + "') AND (time_window[1].date = DATE '2016-10-"+str(day)+"') order by time_window;"
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
                           print("FORECAST : ", predictions[route[0], route[1], day,momento_del_dia][((lhs-rhs)/1200).seconds])
                           print("ROWS2 : ",rows2[0][1])
                        y_test_sum += abs((rows2[0][1] - predictions[route[0], route[1], day,momento_del_dia][((lhs-rhs)/1200).seconds]) / rows2[0][1])
                        count += 1
           intervals_sum += y_test_sum/count;     
        routes_sum += intervals_sum /len(time_intervals)
print("Error MAPE : ", (routes_sum/len(routes)))






















