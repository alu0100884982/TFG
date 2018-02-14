import psycopg2
import pandas as pd
from pandas import set_option
from numpy import loadtxt
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import math
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error

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
set_option( 'display.width' , 100)

# split data into X and y
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(travel_time_dataframe.iloc[:,0])
X_1 = label_encoder.transform(travel_time_dataframe.iloc[:,0])
X_2 = travel_time_dataframe.iloc[:,1]
X_3 = [float(value[1].time().strftime("%H.%M")) for value in travel_time_dataframe.iloc[:,2]]
X_train = pd.concat([pd.DataFrame(X_1, columns=['intersection_id']),pd.DataFrame(X_2),pd.DataFrame(X_3, columns=['hour'])], axis = 1)
y_train = travel_time_dataframe.iloc[:,3]
model = XGBRegressor()
model.fit(X_train, y_train)


try:
    conn = psycopg2.connect("dbname='tfgtest1' user='javisunami' host='localhost' password='javier123'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""SELECT *
FROM travel_time_intersection_to_tollgate_test1 ;""")
rows = cur.fetchall()
colnames = ['intersection_id','tollgate_id','time_window','avg_travel_time']
data_test = pd.DataFrame(rows, columns=colnames)
set_option( 'display.width' , 100)
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(data_test.iloc[:,0])
X_1_test = label_encoder.transform(data_test.iloc[:,0])
X_2_test = data_test.iloc[:,1]
X_3_test = [float(value[1].time().strftime("%H.%M")) for value in data_test.iloc[:,2]]
X_test = pd.concat([pd.DataFrame(X_1_test, columns=['intersection_id']),pd.DataFrame(X_2_test),pd.DataFrame(X_3_test, columns=['hour'])], axis = 1)
y_test = data_test.iloc[:,3]


#make predictions for test data
y_pred = model.predict(X_test)
print(mean_squared_error(y_test, y_pred))
#print(pd.DataFrame(y_pred))
#print(y_test)
#rows = y_test.size
#print(sum(np.absolute(pd.DataFrame(y_test) - pd.DataFrame(y_pred)))/rows)
# evaluate predictions
#accuracy = model.score(y_test, pd.DataFrame(y_pred))
#print(accuracy)
