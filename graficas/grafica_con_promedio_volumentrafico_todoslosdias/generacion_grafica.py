import psycopg2
import matplotlib.pyplot as plt
import collections as coll
import matplotlib.dates as dates
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""SELECT * 
FROM traffic_volume_tollgates_modified
ORDER BY tollgate_id, time_window;""")
rows = cur.fetchall()
dates = sorted(set([row[1][1].date().strftime('%Y-%m-%d') for row in rows]))
dates.pop()
tollgates = [1,2,3]
directions = [0,1]
keys = {}

for tollgate in tollgates:
        for date in dates:
                for direction in directions:
                   keys[(tollgate,date,direction)] = []

counter = 0;
for row in rows:
        if (row[1][1].date().strftime('%Y-%m-%d') != '2016-10-18'):
               keys[(row[0], row[1][1].date().strftime('%Y-%m-%d'), row[2] )].append(row[3])       
keys = coll.OrderedDict(sorted(keys.items()))
fontP = FontProperties()
fontP.set_size('small')
counter = 1


# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('graficas_pdf.pdf') as pdf:
     for direction in directions:
        values = [[] for _ in range(len(tollgates) - direction)]
        for date in dates:
             for idx, tollgate in enumerate(tollgates):
                if(direction == 1 and tollgate == 2):
                        pass
                elif(direction == 1 and tollgate == 3):
                      values[idx - 1].append(sum(keys[(tollgate, date, direction)])/len(keys[(tollgate, date, direction)]))
                else :
                       values[idx].append(sum(keys[(tollgate, date, direction)])/len(keys[(tollgate, date, direction)]))
                                    
        x = np.array(range(len(dates)))     
        for idx, tollgate in enumerate(tollgates):
           if(direction == 1 and tollgate == 2):
              pass;
           else:
              if (direction == 1 and tollgate == 3):
                  idx = idx - 1;
              plt.plot(x,values[idx], label = str(tollgate))
              plt.xticks(x, dates, rotation="vertical")
              fig = plt.gcf()
              fig.subplots_adjust(bottom=0.3)
              plt.legend(prop=fontP)
              plt.xlabel('Dates')
              plt.ylabel('Traffic volume')
              plt.ylim(0,160)
              if (direction == 0):
                plt.title("Entrada")
              else:
                plt.title("Salida")
        pdf.savefig()
        plt.close()
 
