import pandas as pd
import numpy as np

# Get dataset and split
dataset = pd.read_csv(r"Dataset/winequality.csv")
X = dataset.iloc[:,0:10].values
y = dataset.iloc[:,11].values

#Scaling the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Splitting into training and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=0, stratify=y)

#Creating the model
from sklearn.svm import SVR
# regressor = SVR(kernel = 'linear')
# regressor = SVR(kernel = 'poly')
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train)

#Prediction for test data
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error
print(f"Training Score: {regressor.score(X_train,y_train)}")
print(f"R Squared Value: {r2_score(y_test,y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test,y_pred)}")