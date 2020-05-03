import pandas as pd
import numpy as np

# Get dataset and split
dataset = pd.read_csv(r"Datasets/train.csv")
dataset = pd.get_dummies(dataset,columns=["price_range"],prefix=["price_range"])
X = dataset.iloc[:,0:-4].values
y = dataset.iloc[:,-4:].values
y_columns = dataset.iloc[:,-4:].columns

# Standardise the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


#Splitting into training and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=0, stratify=y)

import keras
from keras.models import Sequential
from keras.layers import Dense

#Create Neural Network model
model = Sequential()
model.add(Dense(activation="relu", input_dim=X_train.shape[1], units=32))
model.add(Dense(activation="relu", units=64))
model.add(Dense(activation="relu", units=64))
model.add(Dense(activation="softmax", units=4))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=100, batch_size=64)


y_pred = model.predict(X_test)

pred = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
test = [np.argmax(y_test[i]) for i in range(len(y_test))]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test, pred)


from sklearn.metrics import classification_report


testCombo = sc.fit_transform(np.array([
  [1021,1,0.5,1,0,1,53,0.7,136,3,6,905,1988,2631,17,3,7,1,1,0],
  [563,1,0.5,1,2,1,41,0.9,145,5,6,1263,1716,2603,11,2,9,1,1,0],
  [615,1,2.5,0,0,0,10,0.8,131,6,9,1216,1786,2769,16,8,11,1,0,0],
  [1821,1,1.2,0,13,1,44,0.6,141,2,14,1208,1212,1411,8,2,15,1,1,0],
  [1859,0,0.5,1,3,0,22,0.7,164,1,7,1004,1654,1067,17,1,10,1,0,0],
  [1821,0,1.7,0,4,1,10,0.8,139,8,10,381,1018,3220,13,8,18,1,0,1],
  [1954,0,0.5,1,0,0,24,0.8,187,4,0,512,1149,700,16,3,5,1,1,1],
  [1445,1,0.5,0,0,0,53,0.7,174,7,14,386,836,1099,17,1,20,1,0,0],
  [509,1,0.6,1,2,1,9,0.1,93,5,15,1137,1224,513,19,10,12,1,0,0],
  [769,1,2.9,1,0,0,9,0.1,182,5,1,248,874,3946,5,2,7,0,0,0],
  [1520,1,2.2,0,5,1,33,0.5,177,8,18,151,1005,3826,14,9,13,1,1,1],
  [1815,0,2.8,0,2,0,33,0.6,159,4,17,607,748,1482,18,0,2,1,0,0],
  [803,1,2.1,0,7,0,17,1.0,198,4,11,344,1440,2680,7,1,4,1,0,1],
  [1866,0,0.5,0,13,1,52,0.7,185,1,17,356,563,373,14,9,3,1,0,1],
  [775,0,1.0,0,3,0,46,0.7,159,2,16,862,1864,568,17,15,11,1,1,1],
  [838,0,0.5,0,1,1,13,0.1,196,8,4,984,1850,3554,10,9,19,1,0,1],
  [595,0,0.9,1,7,1,23,0.1,121,3,17,441,810,3752,10,2,18,1,1,0],
  [1131,1,0.5,1,11,0,49,0.6,101,5,18,658,878,1835,19,13,16,1,1,0],
  [682,1,0.5,0,4,0,19,1.0,121,4,11,902,1064,2337,11,1,18,0,1,1],
  [772,0,1.1,1,12,0,39,0.8,81,7,14,1314,1854,2819,17,15,3,1,1,0],
  [1709,1,2.1,0,1,0,13,1.0,156,2,2,974,1385,3283,17,1,15,1,0,0],
  [1949,0,2.6,1,4,0,47,0.3,199,4,7,407,822,1433,11,5,20,0,0,1],
  [1602,1,2.8,1,4,1,38,0.7,114,3,20,466,788,1037,8,7,20,1,0,0],
  [503,0,1.2,1,5,1,8,0.4,111,3,13,201,1245,2583,11,0,12,1,0,0],
  [961,1,1.4,1,0,1,57,0.6,114,8,3,291,1434,2782,18,9,7,1,1,1],
  [519,1,1.6,1,7,1,51,0.3,132,4,19,550,645,3763,16,1,4,1,0,1],
  [956,0,0.5,0,1,1,41,1.0,143,7,6,511,1075,3286,17,8,12,1,1,0],
  [1453,0,1.6,1,12,1,52,0.3,96,2,18,187,1311,2373,10,1,10,1,1,1]
]))

predictions = (np.round(model.predict(testCombo),2))
predictions = [np.where(l == np.max(l))[0][0] for l in predictions]
predictions = np.array(predictions)


print(cm)
print(predictions)
print(classification_report(test,pred,digits=3))

#print(y_columns.values)

