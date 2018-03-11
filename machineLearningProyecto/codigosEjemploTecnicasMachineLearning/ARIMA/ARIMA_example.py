from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting.autocorrelation_plot import autocorrelation_plot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m') #The method strptime() parses a string representing a time according to a format.

series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
print(series)
#series.plot()
#pyplot.show()
autocorrelation_plot(series)
pyplot.show()
