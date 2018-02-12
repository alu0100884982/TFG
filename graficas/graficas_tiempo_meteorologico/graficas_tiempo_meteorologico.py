import psycopg2
import matplotlib.pyplot as plt
import collections as coll
import matplotlib.dates as dates
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

def create_graphic(plt,values,attributes,dates):
         hours = [value[0] for value in values]
         for i in list(range(7)): 
                plt.plot(hours,[value[1][i] for value in values], label = attributes[i])
                plt.legend(prop=fontP)
         plt.title(dates[index])
         plt.xlabel('Hours')
         plt.ylabel('Quantity')
         pdf.savefig()
         plt.close()     
         return


try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""SELECT *
FROM weather_data_modified;""")
rows = cur.fetchall()
dates = sorted(set([row[0].strftime('%Y-%m-%d') for row in rows]))
hours = sorted(set([row[1] for row in rows]))
fontP = FontProperties()
fontP.set_size('small')
index = 0;
values = []
attributes = ['pressure (hPa)', 'sea_pressure(hPa)', 'wind_direction(º)', 'wind_speed(m/s)', 'temperature(ºC)',
              'rel_humidity', 'precipitation(mm)']
with PdfPages('graficas_pdf.pdf') as pdf:
        for row in rows:
             if (dates[index] == row[0].strftime('%Y-%m-%d')) :
                print("Row : ", row)
                values.append([row[1],row[2:]])
             else:
                 create_graphic(plt,values,attributes,dates)
                 values = []
                 values.append([row[1],row[2:]])
                 index = index + 1;
                 print(dates[index])
        create_graphic(plt,values,attributes,dates)

