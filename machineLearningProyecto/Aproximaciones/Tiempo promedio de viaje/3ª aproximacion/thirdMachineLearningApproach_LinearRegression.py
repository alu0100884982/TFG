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
from sklearn.ensemble import RandomForestRegressor
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
for i in range(len(y_pred)):
   print("PRED: ", y_pred[i], "  REAL: ", y_test[i])
print("ERROR : ", errores_predicciones)
################################################################################################################################################






