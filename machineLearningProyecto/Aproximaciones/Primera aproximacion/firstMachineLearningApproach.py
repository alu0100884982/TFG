import psycopg2
import pandas as pd
from pandas import set_option
from numpy import loadtxt
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import math
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error

try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""select pressure, sea_pressure, wind_direction, wind_speed, temperature, rel_humidity, precipitation, two_hours_previous, avg_travel_time
from c_1_08_00_08_20;""")
rows = cur.fetchall() #67 instancias
colnames = ['pressure', 'sea_pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation', 'two_hours_previous' ,'avg_travel_time']
travel_time_dataframe = pd.DataFrame(rows, columns=colnames)
set_option( 'display.width' , 100)

X_train = travel_time_dataframe.iloc[:, 0:8]
y_train = travel_time_dataframe.iloc[:,8]
model = XGBRegressor()
model.fit(X_train, y_train)

try:
    conn2 = psycopg2.connect("dbname='tfgtest1' user='javisunami' host='localhost' password='javier123'")
except:
   print("I am unable to connect to the database")

cur2 = conn2.cursor()
cur2.execute("""select pressure, sea_pressure, wind_direction, wind_speed, temperature, rel_humidity, precipitation, two_hours_previous
from vista_travel_time_contiempometeorologico
where intersection_id = 'C' AND tollgate_id = '1' AND time_window[1].time = '8:00';""")
rows2 = cur2.fetchall()
colnames = ['pressure', 'sea_pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation', 'two_hours_previous']
travel_time_prediction = pd.DataFrame(rows2, columns=colnames)
set_option( 'display.width' , 100)
X_test = travel_time_prediction.iloc[:, 0:8]

#make predictions for test data
y_pred = model.predict(X_test)
print(y_pred)
#print(y_test - y_pred)

