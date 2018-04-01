
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix

wine = pd.read_csv('wine_data.csv', names = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"])
# Deletes the first columns of the table because its values are the type of wine (output value)
X = wine.drop('Cultivator',axis=1)
y = wine['Cultivator']
X_train, X_test, y_train, y_test = train_test_split(X, y)
'''
Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data. Note that you must apply the same scaling to the test set for meaningful results. There are a lot of different methods for normalization of data, we will use the built-in StandardScaler for standardization.
'''
scaler = StandardScaler()
# Fit only to the training data. It computes the mean and std to be used for later scaling.
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
'''
If you do want to extract the MLP weights and biases after training your model, you use its public attributes coefs_ and intercepts_.

coefs_ is a list of weight matrices, where weight matrix at index i represents the weights between layer i and layer i+1.

intercepts_ is a list of bias vectors, where the vector at index i represents the bias values added to layer i+1.
'''
print(mlp.coefs_)









