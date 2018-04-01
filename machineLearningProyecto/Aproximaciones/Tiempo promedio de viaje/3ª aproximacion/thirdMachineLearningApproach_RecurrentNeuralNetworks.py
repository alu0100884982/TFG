#https://blog.statsbot.co/time-series-prediction-using-recurrent-neural-networks-lstms-807fa6ca7f
#http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
import psycopg2
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import datetime
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
	
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
# Normalizes the dataset using the MinMaxScaler preprocessing class from the scikit-learn library.
dates_traveltime = pd.DataFrame(dates_traveltime['avg_travel_time'].values, index=dates_traveltime['date'])
scaler = MinMaxScaler(feature_range=(0, 1))
dates_traveltime = scaler.fit_transform(dates_traveltime)
train_size = int(len(dates_traveltime) * 0.67)
test_size = len(dates_traveltime) - train_size
train, test = dates_traveltime[0:train_size,:], dates_traveltime[train_size:len(dates_traveltime),:]
print(len(train), len(test))
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print("Train X: ", trainX)
