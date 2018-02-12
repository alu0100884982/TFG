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
FROM traffic_volume_tollgates_modified
ORDER BY tollgate_id, time_window;""")
rows = cur.fetchall()
dates = sorted(set([row[1][1].date().strftime('%Y-%m-%d') for row in rows]))
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
               keys[(row[0], row[1][1].date().strftime('%Y-%m-%d'), row[2] )].append([row[1][1].time(), row[3]])
        
keys = coll.OrderedDict(sorted(keys.items()))
fontP = FontProperties()
fontP.set_size('small')
counter = 1

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('graficas_pdf.pdf') as pdf:
        for date in dates:
           fig = plt.figure()
           ax1 = fig.add_subplot(121)
           ax2 = fig.add_subplot(122)
           for tollgate in tollgates:
               hours = [element[0].strftime("%H.%M") for element in keys[(tollgate, date, 0)]]
               values = [element[1] for element in keys[(tollgate, date, 0)]]
               ax1.plot(hours, values, label = str(tollgate))
               hours = [element[0].strftime("%H.%M") for element in keys[(tollgate, date, 1)]]
               values = [element[1] for element in keys[(tollgate, date, 1)]]
               ax2.plot(hours, values, label = str(tollgate))
               plt.show()
               plt.close()       
   #          plt.legend(prop=fontP)
    #         plt.title(date)
    #         plt.xlabel('Hours')
     #        plt.ylabel('Average Travel Time (seconds)')
     #        plt.ylim(0, 1100)
     #        plt.draw();
       #      pdf.savefig()
       
      #       plt.close()
