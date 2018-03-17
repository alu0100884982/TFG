import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from sklearn import model_selection
from numpy import loadtxt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import math
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
import datetime
import re
from matplotlib.backends.backend_pdf import PdfPages

try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""CREATE EXTENSION dblink""")
cur.execute("""SELECT P.*
FROM dblink('dbname=tfgtraining2 port=5432 host=127.0.0.1 user=javisunami password=javier123', 'SELECT * FROM travel_time_intersection_to_tollgate_training2 WHERE (time_window[1].time BETWEEN TIME ''08:00:00'' AND TIME ''09:40:00'') OR (time_window[1].time BETWEEN TIME ''17:00:00'' AND TIME ''18:40:00'') ORDER BY intersection_id, tollgate_id, time_window') 
AS P( intersection_id char(1),
tollgate_id smallint,
time_window varchar(50),
avg_travel_time float);""")
rows = cur.fetchall()
colnames = ['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time']
intervals_to_predict_real_avgtraveltime = pd.DataFrame(rows, columns=colnames)
routes = np.array(intervals_to_predict_real_avgtraveltime.iloc[:,0:2].values.tolist())
time_intervals = np.array(intervals_to_predict_real_avgtraveltime.iloc[:,2].values.tolist())
routes = np.unique(routes, axis=0);
aux = np.array([])
for i,time_interval in enumerate(time_intervals):
        time_interval = re.sub(r'[{\"}]', '', time_interval).split(',')
        time_interval[0] = datetime.datetime.strptime(time_interval[0] , '%Y-%m-%d %H:%M:%S')
        time_interval[1] = datetime.datetime.strptime(time_interval[1] , '%Y-%m-%d %H:%M:%S')
        aux = np.append(aux,(datetime.time(time_interval[0].hour,time_interval[0].minute), datetime.time(time_interval[1].hour,time_interval[1].minute)));
time_intervals = np.unique(aux);
time_intervals = np.delete(time_intervals, -1)
time_intervals = np.delete(time_intervals, 6)
real_avgtraveltime_values = np.array([])
predicted_avgtraveltime_values = np.array([])
colnames = [ 'type_day','twenty_min_previous','forty_min_previous','sixty_min_previous','eighty_min_previous','onehundred_min_previous','onehundredtwenty_min_previous','pressure','sea_pressure','wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation','avg_travel_time']

x = np.array(range(len(time_intervals)))
string_time_intervals = [interval.strftime('%H:%M') for interval in time_intervals]
with PdfPages('graficas_pdf.pdf') as pdf:
   for route in routes:
     num_instances = np.array([])
     for interval in time_intervals:
        conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
        cur = conn.cursor()
        query = "select * from " + route[0].lower() + "_" + str(route[1]) + "_" + str(interval.hour) + "_" + str(interval.minute) + ";"
        cur.execute(query)
        rows = cur.fetchall()
        num_instances = np.append(num_instances,len(rows))
     bars = plt.bar(x,num_instances, align='center')
     plt.xticks(x, string_time_intervals, rotation="vertical")
     plt.yticks(np.arange(0, 150, 10))
     fig = plt.gcf()
     fig.subplots_adjust(bottom=0.3)
     plt.xlabel('Intervals')
     plt.ylabel('Number of training instances')
     plt.ylim(0,150)
     plt.title("Ruta " + str(route[0]) + "_" + str(route[1]))
     pdf.savefig()
     plt.close()


       
      
