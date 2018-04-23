#https://blog.statsbot.co/time-series-prediction-using-recurrent-neural-networks-lstms-807fa6ca7f
#http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
#https://machinelearningmastery.com/promise-recurrent-neural-networks-time-series-forecasting/
#https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

# Mejor valor: look_back = 5 (254 de error cuadrático medio)
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

#############Funciones###################################

def timeseries_to_supervised(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff


def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]



###########Preparación de los datos##############################


try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
query = "select time_window[1], avg_travel_time from travel_time_intersection_to_tollgate_modified where intersection_id = 'A' AND tollgate_id = 2 order by time_window;"
cur.execute(query)
rows = cur.fetchall()
dates_traveltime = pd.DataFrame.from_records(rows, columns=['date','avg_travel_time'])

#Rellenamos los datos de entrenamiento con datos faltantes
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

#Diferenciamos la serie para asegurarnos de que la serie temporal es estacionaria.
dates_traveltime_difference = difference(dates_traveltime.values,1)
dates_traveltime_difference = [element[1] for element in dates_traveltime_difference]
dates = dates_traveltime['date'].iloc[1:]
dates_traveltime_differenced = pd.DataFrame(dates_traveltime_difference, index=dates, columns = ['avg_travel_time'])

#Comprobamos que se han realizado correctamente las diferencias
inverted = []
for i in range(len(dates_traveltime_difference)):
        value = inverse_difference(dates_traveltime['avg_travel_time'].values, dates_traveltime_difference[i], len(dates_traveltime.index)-i)
        inverted.append(value)

#Convertimos los valores de la serie temporal a una estructura de aprendizaje supervisado
look_back = 5
supervised_values = timeseries_to_supervised(dates_traveltime_differenced['avg_travel_time'].values, look_back)
for i in range(len(supervised_values)):
      print("Supervised values : ", supervised_values[i])

#Escalamos los valores de tiempo promedio de viaje al rango [-1,1] debido a que la función de activación para LSTM por defecto es tanh.
train_size = int(len(dates_traveltime) * 0.67)
test_size = len(dates_traveltime) - train_size
train, test = supervised_values[0:train_size], supervised_values[train_size:len(supervised_values)]
scaler = MinMaxScaler(feature_range=(-1, 1))
train = pd.DataFrame(train, columns= ['1','2','3','4','5'])
print("TRAIN : ", train)
train = scaler.fit(train)


'''

trainX, trainY = timeseries_to_supervised(train, look_back)
testX, testY = timeseries_to_supervised(test, look_back)
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(5, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(50):
        model.fit(trainX, trainY, epochs=1, batch_size=20, verbose=0, shuffle=False)
        model.reset_states()

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
trainScore = mean_squared_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f RMSE' % (trainScore))
testScore = mean_squared_error(testY[0], testPredict[:,0])
print('Test Score: %.2f RMSE' % (testScore))
trainPredictPlot = numpy.empty_like(dates_traveltime)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dates_traveltime)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dates_traveltime)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dates_traveltime))
#plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
'''

