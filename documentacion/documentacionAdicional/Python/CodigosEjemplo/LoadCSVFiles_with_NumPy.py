# Load CSV using NumPy
from numpy import loadtxt
filename = 'datasets/pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rb' )
data = loadtxt(raw_data, delimiter=",")
print(data)
print(data.shape)
