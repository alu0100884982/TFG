import psycopg2
import pandas as pd
from pandas import set_option
from numpy import loadtxt
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Predicciones sobre el conjunto de entrenamiento
try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""select time_window[1], avg_travel_time
from travel_time_intersection_to_tollgate_modified 
where intersection_id = 'A' AND tollgate_id = 2 AND (time_window[1].date BETWEEN DATE '2016-07-25' AND DATE '2016-07-31') 
order by time_window;""")
rows = cur.fetchall()
df = pd.DataFrame.from_records(rows, columns=['date','avg_travel_time'])
df = df[(df.avg_travel_time > 50) & (df.avg_travel_time <= 150)]
serie = pd.Series(df['avg_travel_time'].values, index=df['date'].values)

with PdfPages('graficas_pdf.pdf') as pdf:
     serie.plot()
     plt.xlabel('Intervalos')
     plt.ylabel('Tiempo promedio de viaje')
     plt.ylim(0,170)
     plt.title(" Del 2016-07-25 al 2016-07-31 por horas en la ruta A-2")
     fig = plt.gcf()
     fig.subplots_adjust(bottom=0.4)
     pdf.savefig()
     plt.close()

cur = conn.cursor()
cur.execute("""select time_window[1], avg_travel_time
from travel_time_intersection_to_tollgate_modified 
where intersection_id = 'A' AND tollgate_id = 2 AND time_window[1].time = TIME '09:40:00'
""")
rows = cur.fetchall()
df = pd.DataFrame.from_records(rows, columns=['date','avg_travel_time'])
df = df[(df.avg_travel_time > 50) & (df.avg_travel_time <= 150)]
serie = pd.Series(df['avg_travel_time'].values, index=df['date'].values)

with PdfPages('graficas_pdf_unahora_diasentrenamiento.pdf') as pdf:     
     serie.plot()
     plt.xlabel('Intervalos')
     plt.ylabel('Tiempo promedio de viaje')
     plt.ylim(0,160)
     plt.title("Todos los días de entrenamiento a las 9:40 en la ruta A-2")
     fig = plt.gcf()
     fig.subplots_adjust(bottom=0.4)
     pdf.savefig()
     plt.close()
       
      
