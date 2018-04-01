# https://machinelearningmastery.com/time-series-forecasting-supervised-learning/ -> Ver los enlaces del final sobre series temporales.
# https://www.youtube.com/watch?v=SSu00IRRraY
# https://www.sciencedirect.com/science/article/abs/pii/S0169743903001114 


import psycopg2
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.svm import SVR
import datetime
import matplotlib.pyplot as plt

# Prueba de regresión lineal, teniendo en cuenta los atributos del volumen de tráfico y la proporción de coches con etc como columnas de X y el tiempo promedio de viaje.
################################################################################################################################################
try:
   conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
   print("I am unable to connect to the database")


cur = conn.cursor()
cur.execute("""select sum_etc, sum_volume, avg_travel_time
from travel_time_intersection_to_tollgate_modified as left_table NATURAL JOIN (select tollgate_id, time_window,sum(proportion_hasetc_vehicles) as sum_etc, sum(volume) as sum_volume
from traffic_volume_tollgates_modified 
group by tollgate_id, time_window) as traffic;
""")
rows = cur.fetchall()
colnames = ['sum_etc','sum_volume', 'avg_travel_time']
volume_traveltime = pd.DataFrame(rows, columns=colnames)
print("Coeficiente de Pearson entre la suma de volumen y el tiempo promedio de viaje: ", np.corrcoef(volume_traveltime['sum_volume'], volume_traveltime['avg_travel_time'] ))
array = volume_traveltime.values
X = array[:,0:2]
Y = array[:,2]
test_size = 0.33
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
errores_predicciones = mean_squared_error(y_pred, y_test)
#for i in range(len(y_pred)):
    #    print("PRED: ", y_pred[i], "  REAL: ", y_test[i])
print("ERROR : ", errores_predicciones)
################################################################################################################################################

# Prueba de SVR
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
dates_traveltime = pd.DataFrame.from_records(rows, columns=['date','avg_travel_time'])
array = dates_traveltime.values
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

# Intentar hacer lo de time series to supervised learning.

