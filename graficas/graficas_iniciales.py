
import psycopg2
import matplotlib.pyplot as plt
import collections as coll
import matplotlib.dates as dates

try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""SELECT *
FROM travel_time_intersection_to_tollgate_modified ;""")
rows = cur.fetchall()
dates = sorted(set([date[2][1].date().strftime('%Y-%m-%d') for date in rows]))
pairs=[['B',1],['B',3],['A',2],['A',3],['C',1],['C',3]]
pairs = sorted(pairs);
keys = {}


for date in dates:
        for pair in pairs:
                keys[(pair[0],pair[1],date)] = []

counter = 0;
for row in rows:
        keys[(row[0], row[1], row[2][1].date().strftime('%Y-%m-%d') )].append([row[2][1].time(), row[3]])
        
keys = coll.OrderedDict(sorted(keys.items()))

for element in pairs:
   horas = []
   valores = []
   for key, value in keys.items() :
        
        if (key[0] == element[0] and key[1] == element[1]):
            horas = [pair[0].strftime("%H.%M") for pair in value]
            valores = [pair[1] for pair in value];
            print([float(hora) for hora in horas])
            plt.plot(horas, valores); 
   print("Intersección: ", element[0], " Tollgate : ", element[1]);
   plt.show();  
#plt.title('Julio de 2016')
#plt.ylabel('Tiempo medio de viaje')
#plt.xlabel('Momento del día')
