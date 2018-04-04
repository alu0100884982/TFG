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
# Prueba de SVR con las columnas date-avg_travel_time
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
print("HEAD : ", dataframe.head())
array = dataframe.values
X = array[:,0:6]
y = array[:,6]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=7)
mlp = MLPRegressor(hidden_layer_sizes=(13,13,13),max_iter=900)
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
