import psycopg2
import matplotlib.pyplot as plt
import pandas as pd
import collections as coll
import matplotlib.dates as dates
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""SELECT *
FROM travel_time_intersection_to_tollgate_modified order by time_window;""")
rows = cur.fetchall()
dates = sorted(set([date[2][1].date().strftime('%Y-%m-%d') for date in rows]))
dates.pop()
pairs=[['B',1],['B',3],['A',2],['A',3],['C',1],['C',3]]
pairs = sorted(pairs);
keys = {}


for date in dates:
        for pair in pairs:
                keys[(pair[0],pair[1],date)] = []

counter = 0;
for row in rows:
                keys[(row[0], row[1], row[2][0].date().strftime('%Y-%m-%d') )].append([row[2][0].time(), row[3]])
        
keys = coll.OrderedDict(sorted(keys.items()))
fontP = FontProperties()
fontP.set_size('small')
counter = 1
dates = dates[0:2]
# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('graficas_pdf.pdf') as pdf:
      for pair in pairs:
        for date in dates:
                hours = [element[0].strftime("%H:%M") for element in keys[(pair[0], pair[1], date)]]
                values = [element[1] for element in keys[(pair[0], pair[1], date)]]
                serie = pd.Series(values, index=hours)
                serie.plot()
                plt.legend(prop=fontP)
        plt.title(pair)
        plt.xlabel('Hours')
        plt.ylabel('Average Travel Time (seconds)')
        plt.xticks(rotation=90)
        plt.ylim(0, 400)
        plt.draw();
        pdf.savefig()
        plt.close()
