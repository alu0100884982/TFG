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
array = dates_traveltime.values
X = array[:,0]
X = [elemento.strftime('%H%M%S') for elemento in X] #Cojo solo la hora para que el algoritmo SVR realice mejor la predicción (repetición de patrones a unas horas determinadas).
Y = array[:,1]
svr_lin = SVR(kernel = 'linear', C=1e3)
svr_poly = SVR(kernel = 'poly', C=1e3, degree = 2)
svr_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1)
test_size = 0.33
seed = 7
#svr_lin.fit(np.array(X_train).reshape(-1, 1), y_train)
#svr_poly.fit(np.array(X_train).reshape(-1, 1), y_train)
svr_rbf.fit(np.array(X).reshape(-1, 1), Y)


try:
    conn = psycopg2.connect("dbname='tfgtest1' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
query = "select time_window[1], avg_travel_time from travel_time_intersection_to_tollgate_test1  where intersection_id = 'A' AND tollgate_id = 2 order by time_window;"
cur.execute(query)
rows = cur.fetchall()
dates_traveltime2 = pd.DataFrame.from_records(rows, columns=['date','avg_travel_time'])
array = dates_traveltime2.values
X_test = array[:,0]
X_test = [elemento.strftime('%H%M%S') for elemento in X_test] #Cojo solo la hora para que el algoritmo SVR realice mejor la predicción (repetición de patrones a unas horas determinadas).
y_test = array[:,1]
predictions = svr_rbf.predict(np.array(X_test).reshape(-1, 1))
for i in range(len(predictions)):
   print("PRED: ", predictions[i], " REAL : ", y_test[i])
print("ERROR : ", mean_squared_error(predictions, y_test))
indexes = [i for i in range(len(X_test))]
plt.plot(indexes,predictions, color='blue')
plt.plot(indexes,y_test, color='black')
plt.show()

#############################################################################################################################################
# Prueba de SVR con las columnas date-avg_travel_time => Time Series to Supervised Learning.
series = pd.Series(dates_traveltime['avg_travel_time'].values, index=dates_traveltime['date'])
dataframe = pd.DataFrame()
for i in range(9,0,-1):
        dataframe['t-'+str(i)] = series.shift(i)
dataframe['t'] = series.values
dataframe = dataframe[10:]
array = dataframe.values
X = array[:,0:9]
y = array[:,9]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=7)
svr_rbf = SVR(kernel='rbf',C=1e3,gamma='auto')
svr_rbf.fit(X_train, y_train)
#print("X_test : ", X_test)
predictions = svr_rbf.predict(X_test)
#for i in range(len(predictions)):
 #  print("PREDICTION: ", predictions[i], " REAL : ", y_test[i])
print("ERRORR1 : ", mean_squared_error(predictions, y_test))
indexes = [i for i in range(len(X_test))]
plt.plot(indexes,predictions, color='blue')
plt.plot(indexes,y_test, color='black')
plt.show()

mlp = MLPRegressor(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
for i in range(len(predictions)):
 print("PREDICTION: ", predictions[i], " REAL : ", y_test[i])
print("ERRORR2 : ", mean_squared_error(predictions, y_test))
'''
X = array[:,0:-1]
y = array[:,-1]
# fit random forest model
model = RandomForestRegressor(n_estimators=500, random_state=1)
model.fit(X, y)
# show importance scores
print(model.feature_importances_)
# plot importance scores

names = dataframe.columns.values[0:-1]
ticks = [i for i in range(len(names))]
plt.bar(ticks, model.feature_importances_)
plt.xticks(ticks, names)
plt.show()
'''
# http://ieeexplore.ieee.org/document/4708961/ => It is important to choose good parameters in Support Vector Regression (SVR) modeling. Choosing different parameters will influence the accuracy of SVR models. This paper proposes a parameter choosing method of SVR models for time series prediction. In the light of data features of time series, the paper improves the traditional Cross-Validation method, and combines the improved Cross-Validation with epsilon-weighed SVR in order to get good parameters of models. The experiments show that the method is effective for time series prediction.
