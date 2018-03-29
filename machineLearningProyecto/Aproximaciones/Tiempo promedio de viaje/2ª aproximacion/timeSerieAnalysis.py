import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
import datetime;
# Predicciones sobre el conjunto de entrenamiento
def generacionGrafica(ruta, ruta_array):
        try:
            conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
        except:
            print("I am unable to connect to the database")
        cur = conn.cursor()
        query = "select time_window[1], avg_travel_time from travel_time_intersection_to_tollgate_modified where " + ruta + " order by time_window;"
        cur.execute(query)
        rows = cur.fetchall()
        df = pd.DataFrame.from_records(rows, columns=['date','avg_travel_time'])
        minimum_date = min(df.date)
        maximum_date = max(df.date)
        date_aux = minimum_date
        while (date_aux != maximum_date):
               if (date_aux not in df['date']):
                 valores_avg_travel = []
                 for row in df.values:
                        if (row[0].time() == date_aux.time()):
                                valores_avg_travel.append(row[1])
                 df.loc[len(df)] = [date_aux, np.mean(valores_avg_travel)]
               date_aux += datetime.timedelta(minutes=20) 
        print("DATES : ", df)
        df = df.sort_index()
        serie = pd.Series(df['avg_travel_time'].values, index=df['date'].values)
        serie.plot()
        plt.xlabel('Intervalos')
        plt.ylabel('Tiempo promedio de viaje')
        plt.ylim(0,170)
        plt.title(" Del 2016-07-25 al 2016-07-31 por horas en la ruta " + str(ruta_array))
        plt.show()
        plt.close()
        new_df = pd.DataFrame(data=df['avg_travel_time'].values, index = df['date'].values);
        print(new_df.index)
        # Another tool to visualize the data is the seasonal_decompose function in statsmodel. With this, the trend and seasonality become even more obvious.
        decomposition = seasonal_decompose(new_df,freq=72)
        fig = decomposition.plot()  
        plt.show()
        plt.close()
        X = serie.values
        split = int(len(X)/ 2)
        X1, X2 = X[0:split], X[split:]
        mean1, mean2 = X1.mean(), X2.mean()
        var1, var2 = X1.var(), X2.var()
        print('mean1=%f, mean2=%f' % (mean1, mean2))
        print('variance1=%f, variance2=%f' % (var1, var2))
        #Augmented Dickey-Fuller test
        result = adfuller(X)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
	        print('\t%s: %.3f' % (key, value))


rutas = [('A',2), ('A',3), ('B',1), ('B',3), ('C',1), ('C',3)]

def generacionAutocorrelacion(ruta):
        try:
            conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
        except:
            print("I am unable to connect to the database")
        cur = conn.cursor()
        query = "select time_window[1], avg_travel_time from travel_time_intersection_to_tollgate_modified where " + ruta + "order by time_window;"
        cur.execute(query)
        rows = cur.fetchall()
        df = pd.DataFrame.from_records(rows, columns=['date','avg_travel_time'])
        serie = pd.Series(df['avg_travel_time'].values, index=df['date'].values)
        autocorrelation_plot(serie)
        plt.show()
        plt.close()

for ruta in rutas:
       generacionGrafica("intersection_id = '" + str(ruta[0]) + "' AND tollgate_id = " + str(ruta[1]), ruta)
       generacionAutocorrelacion("intersection_id = '" + str(ruta[0]) + "' AND tollgate_id = " + str(ruta[1]))
       

 
