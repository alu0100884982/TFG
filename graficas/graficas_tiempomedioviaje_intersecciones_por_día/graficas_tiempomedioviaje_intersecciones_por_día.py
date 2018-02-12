import psycopg2
import matplotlib.pyplot as plt
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
        if (row[2][1].date().strftime('%Y-%m-%d') != '2016-10-18'):
                keys[(row[0], row[1], row[2][1].date().strftime('%Y-%m-%d') )].append([row[2][1].time(), row[3]])
        
keys = coll.OrderedDict(sorted(keys.items()))
fontP = FontProperties()
fontP.set_size('small')
counter = 1

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('graficas_pdf.pdf') as pdf:
        for date in dates:
             for pair in pairs:
                hours = [element[0].strftime("%H.%M") for element in keys[(pair[0], pair[1], date)]]
                values = [element[1] for element in keys[(pair[0], pair[1], date)]]
                plt.plot(hours,values, label = str(pair))
                plt.legend(prop=fontP)
             plt.title(date)
             plt.xlabel('Hours')
             plt.ylabel('Average Travel Time (seconds)')
             plt.ylim(0, 1100)
             plt.draw();
             pdf.savefig()
             plt.close()
         
