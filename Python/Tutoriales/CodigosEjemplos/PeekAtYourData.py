# View first 20 rows
from pandas import read_csv
filename = "datasets/pima-indians-diabetes.data.csv"
names = [ 'preg' , 'plas' , 'pres' , 'skin' , 'test' , 'mass' , 'pedi' , 'age' , 'class' ]
data = read_csv(filename, names=names)
peek = data.head(20)
print(peek)
