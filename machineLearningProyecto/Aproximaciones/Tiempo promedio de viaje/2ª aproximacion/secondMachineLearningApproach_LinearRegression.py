import psycopg2
import pandas as pd
from pandas import set_option
from numpy import loadtxt
try:
    conn = psycopg2.connect("dbname='tfgdatosmodificados' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""select *
from  a_2_9_40;""")
rows = cur.fetchall()
