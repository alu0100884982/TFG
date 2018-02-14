import psycopg2
import pandas as pd


try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""SELECT *
FROM travel_time_intersection_to_tollgate_modified ;""")
rows = cur.fetchall()
colnames = ['intersection_id','tollgate_id','time_window','avg_travel_time']
travel_time_dataframe = pd.DataFrame(rows, columns=colnames)
print(travel_time_dataframe)
