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
FROM traffic_volume_tollgates_modified order by time_window;""")
rows = cur.fetchall()
dates = sorted(set([date[1][0].date().strftime('%Y-%m-%d') for date in rows]))
pairs=[[1,0],[2,0],[3,0],[1,1],[3,1]]
pairs = sorted(pairs);
keys = {}

for date in dates:
        for pair in pairs:
                keys[(pair[0],pair[1],date)] = []

counter = 0;
for row in rows:
                keys[(row[0], row[2], row[1][0].date().strftime('%Y-%m-%d') )].append([row[1][0].time(), row[3]])
                
keys = coll.OrderedDict(sorted(keys.items()))
counter = 1

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('graficas_pdf.pdf') as pdf:
      for pair in pairs:
        for date in dates:
        
                hours = [element[0].strftime("%H:%M") for element in keys[(pair[0], pair[1], date)]]
                values = [element[1] for element in keys[(pair[0], pair[1], date)]]
                serie = pd.Series(values, index=hours)
                serie.plot(label=date)
                plt.legend(loc=2, prop={'size': 6})
                
        plt.title(pair)
        plt.xlabel('Hours')
        plt.ylabel('Traffic volume')
        plt.xticks(rotation=90)
        plt.ylim(0, 400)
        plt.draw();
        pdf.savefig()
        plt.close()
