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
from matplotlib import pyplot
from xgboost.sklearn import XGBRegressor

#############Funciones###################################

def timeseries_to_supervised(dataset, look_back=1):
    data = []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back+1)]
        data.append(a)
    return np.array(data)


def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff


def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X)
	return yhat[0,0]

def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]


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
#dates_traveltime = dates_traveltime[(dates_traveltime['avg_travel_time'] > 30) & (dates_traveltime['avg_travel_time'] < 150)]
#dates_traveltime = dates_traveltime.reset_index(drop=True)
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
supervised_values= timeseries_to_supervised(dates_traveltime['avg_travel_time'].values, look_back)


#Escalamos los valores de tiempo promedio de viaje al rango [-1,1] debido a que la función de activación para LSTM por defecto es tanh.
scaler = MinMaxScaler(feature_range=(-1, 1))
supervised_values_=scaler.fit_transform(supervised_values)
train_size = int(len(supervised_values_) * 0.67)
test_size = len(supervised_values_) - train_size
train_escalado, test_escalado = supervised_values_[0:train_size], supervised_values_[train_size:len(supervised_values)]


#Ajustamos una red neuronal LSTM a los datos de entrenamiento
X, y = train_escalado[:, 0:-1], train_escalado[:, -1]
print("Y: ", y)
X = X.reshape(X.shape[0], 1, X.shape[1])
nb_epoch = 1
batch_size = 1
neurons = 4
model = Sequential()
model.add(LSTM(neurons, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(nb_epoch):
	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
	model.reset_states()


#Predicciones
predictions = list()
expected_values = list()
contador = 0
suma = 0
'''
for i in range(len(test_escalado)):
	# make one-step forecast
	X, y = test_escalado[i, 0:-1], test_escalado[i, -1]
	yhat = forecast_lstm(model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	#yhat = inverse_difference(dates_traveltime['avg_travel_time'].values, yhat, len(test_escalado)-i+1)
	# store forecast
	predictions.append(yhat)
	expected = dates_traveltime['avg_travel_time'].values[len(train_escalado) + i]
	expected_values.append(expected)
	print('Row=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
	suma += abs(yhat-expected)/expected
	contador += 1
print("RESULTADO : ", (suma/contador))
pyplot.plot(predictions, color="red")
pyplot.plot(expected_values, color="blue")
pyplot.show()
'''
##############################################################3

train, test= supervised_values[0:train_size], supervised_values[train_size:len(supervised_values)]
X_train, y_train = train[:,0:-1], train[:,-1]
X_test, y_test = test[:,0:-1], test[:,-1]
print("Y_TEST : ", test[:,-1])
model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
suma = 0
for i in range(len(y_test)):
     print("REAL : ", y_test[i], " PREDICTED: ", y_pred[i])
     suma += abs(y_pred[i]-y_test[i])/y_test[i]

print("ERROR : ", (suma/len(y_test)))
pyplot.plot(y_pred, color="red")
pyplot.plot(y_test, color="blue")



pyplot.show()
