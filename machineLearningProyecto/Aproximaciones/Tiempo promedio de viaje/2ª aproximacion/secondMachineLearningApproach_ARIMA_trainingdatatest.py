import psycopg2
import pandas as pd
from pandas import set_option
from numpy import loadtxt
from pandas import read_csv
from pandas import datetime
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# Predicciones sobre el conjunto de entrenamiento
try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""select time_window[1], avg_travel_time
from travel_time_intersection_to_tollgate_modified 
where intersection_id = 'A' AND tollgate_id = 2
order by time_window;""")
rows = cur.fetchall()
df = pd.DataFrame.from_records(rows, columns=['date','avg_travel_time'])
df = df[(df.avg_travel_time > 50) & (df.avg_travel_time < 150)]
indexes = [i for i in range(len(df))]
print(df.shape)
dates = pd.DatetimeIndex(df['date'].values)
plt.plot(indexes, df['avg_travel_time'].values)
plt.xticks(indexes, dates.time, rotation="vertical")
#plt.show()
serie = pd.Series(df['avg_travel_time'].values, index=df['date'])
autocorrelation_plot(serie)
#plt.show()
model = ARIMA(serie, order=(3,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
#plt.show()


X = serie.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(9,0,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
