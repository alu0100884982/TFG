
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
intersections = sorted(set([item[0] for item in rows]))
tollgates = set([item[1] for item in rows])
print(dates)
horas = []
valores = []
for row in rows:
       if (row[0] == 'C' and row[1] == 1 and row[2][0].date().strftime('%Y-%m-%d') == "2016-07-19") :
               horas.append(row[2][0]);
               valores.append(row[3]); 
plt.plot(horas,valores)
plt.title('Julio de 2016')
plt.ylabel('Tiempo medio de viaje')
plt.xlabel('Momento del día')
plt.show()
