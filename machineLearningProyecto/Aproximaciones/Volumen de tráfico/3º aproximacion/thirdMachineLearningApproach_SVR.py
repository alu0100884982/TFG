import psycopg2
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.svm import SVR
import datetime
import matplotlib.pyplot as plt
# Prueba de SVR cogiendo como columnas la fecha y el volumen de tráfico y pasándolo a => Time Series to Supervised Learning.
'''
try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
query = "select time_window[1], volume from traffic_volume_tollgates_modified where tollgate_id = 3 and direction = 0 order by time_window;"
cur.execute(query)
rows = cur.fetchall()
dates_trafficvolume = pd.DataFrame.from_records(rows, columns=['date','traffic_volume'])

minimum_date = min(dates_trafficvolume.date)
maximum_date = max(dates_trafficvolume.date)
date_aux = minimum_date
dates_trafficvolume_filled = pd.DataFrame(dates_trafficvolume)
while (date_aux != maximum_date): 
    if (not((date_aux == dates_trafficvolume_filled['date']).any())):
      valores_avg_travel = []
      for row in dates_trafficvolume_filled.values:
        if (row[0].time() == date_aux.time() and row[0].weekday() == date_aux.weekday()):
                valores_avg_travel.append(row[1])
      dates_trafficvolume_filled.loc[len(dates_trafficvolume_filled)] = [date_aux, np.mean(valores_avg_travel)]
    date_aux += datetime.timedelta(minutes=20)
dates_trafficvolume_filled = dates_trafficvolume_filled.sort_index()

series = pd.Series(dates_trafficvolume_filled['traffic_volume'].values, index=dates_trafficvolume_filled['date'])

dataframe = pd.DataFrame()
for i in range(5,0,-1):
        dataframe['t-'+str(i)] = series.shift(i)
dataframe['t'] = series.values
dataframe = dataframe[6:]
print("HEAD1 : ", dataframe.head())
array = dataframe.values
X = array[:,0:5]
y = array[:,5]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=7)
svr_rbf = SVR(kernel='rbf',C=70,gamma=0.00008)
svr_rbf.fit(X_train, y_train)

predictions = svr_rbf.predict(X_test)
predictions = [round(prediction,0) for prediction in predictions]
for i in range(len(predictions)):
   print("PRED: ", predictions[i], " REAL : ", y_test[i])
print("ERROR : ", mean_squared_error(predictions, y_test))
indexes = [i for i in range(len(X_test))]
plt.plot(indexes,predictions, color='blue')
plt.plot(indexes,y_test, color='black')
plt.show()
'''
####Predicción del volumen de tráfico con la técnica de aprendizaje automático denominada SVR.
pairs = [(1,0),(2,0),(3,0),(1,1), (3,1)]
days = list(range(18,25))
intervals_2hours_previous = [(6,8),(15,17)]
intervals_to_predict = ['08:00-10:00','17:00-19:00']
number_intervals_to_predict = 6
predictions = dict()

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
        for day in days:
          for interval in intervals_2hours_previous:
                  minimum_date = min(dates_trafficvolume.date)
                  
                  if (interval[0] == 6):
                   maximum_date = datetime.datetime(2016,10,day,6,0,0)
                  else:
                   maximum_date = datetime.datetime(2016,10,day,15,0,0)
                  date_aux = minimum_date
                  dates_trafficvolume = trafficvolume[dates_traveltime['avg_travel_time'] < 150)]
                  dates_trafficvolume = dates_trafficvolume.reset_index(drop=True)
                  dates_trafficvolume_filled = pd.DataFrame(dates_trafficvolume, index = dates_trafficvolume.index)
                  
                  while (date_aux != maximum_date): 
                    if (not((date_aux == dates_trafficvolume_filled['date']).any())):
                      valores_avg_travel = []
                      for row in dates_trafficvolume_filled.values:
                        if (row[0].time() == date_aux.time() ):
                                valores_avg_travel.append(row[1])
                      dates_trafficvolume_filled.loc[len(dates_trafficvolume_filled)] = [date_aux, np.mean(valores_avg_travel)]
                    date_aux += datetime.timedelta(minutes=20)
                  
                  print("antes: ", dates_trafficvolume_filled)
                  dates_trafficvolume_filled = dates_trafficvolume_filled.sort_values(by='date')
                  print("despues: ", dates_trafficvolume_filled)
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
                  dates_trafficvolume_supervised['t'] = series_dates_trafficvolume_filled .values

                  dates_trafficvolume_supervised = dates_trafficvolume_supervised[number_time_steps_previous:]
                  X_train = dates_trafficvolume_supervised.iloc[:,0:number_time_steps_previous]
                  y_train = dates_trafficvolume_supervised.iloc[:,number_time_steps_previous]
                  svr_rbf = SVR(kernel='rbf',C=10000,gamma=0.00008)
                  svr_rbf.fit(X_train, y_train)
                  
                  previous_row_prediction = dates_trafficvolume_supervised.iloc[-1].shift(-1).values[0:-1]
                  for j in range(number_intervals_to_predict):
                      dataframe_input = pd.DataFrame(previous_row_prediction).T
                     # print("INPUT : ", ((pair[0],pair[1],day,intervals_to_predict[0]) not in predictions.keys()))
                      prediction = round(svr_rbf.predict(dataframe_input)[0], 0)
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
                           
                  for key,val in predictions.items():
                         print(key, "=>", val)



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
         
pairs_sum = 0;   

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
                #print("QUERY : ", query)
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
                           print("FORECAST : ", predictions[pair[0], pair[1], day,momento_del_dia][((lhs-rhs)/1200).seconds])
                           print("ROWS2 : ",rows2[0][1])
                        y_test_sum += abs((rows2[0][1] - predictions[pair[0], pair[1], day,momento_del_dia][((lhs-rhs)/1200).seconds]) / rows2[0][1])
                        count += 1
           intervals_sum += y_test_sum/count;     
        pairs_sum += intervals_sum /len(time_intervals)
print("Error MAPE : ", (pairs_sum/len(pairs)))


