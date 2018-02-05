
import psycopg2
import matplotlib.pyplot as plt;

try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""SELECT *
FROM travel_time_intersection_to_tollgate_modified ;""")
rows = cur.fetchall()
dates = sorted(set([date[2][0].date().strftime('%Y-%m-%d') for date in rows]))
pairs=[['B',1],['B',3],['A',2],['A',3],['C',1],['C',3]]
keys = {}


for date in dates:
        for pair in pairs:
                keys[(pair[0],pair[1],date)] = []

print(keys);
#plt.plot(horas,valores)
#plt.title('Julio de 2016')
#plt.ylabel('Tiempo medio de viaje')
#plt.xlabel('Momento del día')
#plt.show()
