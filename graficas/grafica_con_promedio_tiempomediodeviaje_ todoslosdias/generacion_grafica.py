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
FROM travel_time_intersection_to_tollgate_modified ;""")
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
        if (row[2][0].date().strftime('%Y-%m-%d') != '2016-10-18'):
                keys[(row[0], row[1], row[2][0].date().strftime('%Y-%m-%d') )].append(row[3])
                
print(keys[('A', 2, '2016-09-04')])

keys = coll.OrderedDict(sorted(keys.items()))
fontP = FontProperties()
fontP.set_size('xx-small')
counter = 1

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('graficas_pdf.pdf') as pdf:
        values = [[] for _ in range(len(pairs))]
        for date in dates:
             for idx, pair in enumerate(pairs):
                values[idx].append(sum(keys[(pair[0], pair[1], date)])/len(keys[(pair[0], pair[1], date)])) 
                
        x = np.array(range(len(dates)))
        length = len(x)        
        for idx, pair in enumerate(pairs):
          plt.plot(x[:int((length / 2) - 1)],values[idx][:int((length / 2) - 1)], label = str(pair))
          plt.xticks(x[:int((length / 2) - 1)], dates[:int((length / 2) - 1)], rotation="vertical")
          fig = plt.gcf()
          fig.subplots_adjust(bottom=0.3)
          plt.legend(prop=fontP)
          plt.xlabel('Dates')
          plt.ylabel('Average Travel Time (seconds)')
        pdf.savefig(pad_inches=0.4)
        plt.close()
        
        for idx, pair in enumerate(pairs):
          plt.plot(x[int((length / 2) - 1):],values[idx][int((length / 2) - 1):], label = str(pair))
          plt.xticks(x[int((length / 2) - 1):], dates[int((length / 2) - 1):], rotation="vertical")
          fig = plt.gcf()
          fig.subplots_adjust(bottom=0.3)
          plt.legend(prop=fontP)
          plt.xlabel('Dates')
          plt.ylabel('Average Travel Time (seconds)')
        pdf.savefig(pad_inches=0.4)
        plt.close()
 
