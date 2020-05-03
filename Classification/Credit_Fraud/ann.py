# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'Dataset/creditcard.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # fit then scale, #fit gets the mean and values so scaling is possible
X_test = sc.transform(X_test) #just scale, using the previous fit

#Create class weights
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

print(class_weights)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

import pickle
filepath = r"Saved_Models/fraud_model_ann.pickle"
try:
    classifier = pickle.load(open(filepath,"rb"))
except:

    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(activation="relu", input_dim=X_train.shape[1], units=16, kernel_initializer="uniform"))

    # Adding the second hidden layer
    classifier.add(Dense(activation="relu", units=16, kernel_initializer="uniform"))

    # Adding the output layer
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 16, epochs = 100, class_weight=class_weights)

    pickle.dump(classifier, open(filepath,"wb"))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Ensure that the model only predicts as not fraud if its 99% sure
y_pred = (y_pred > 0.99)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))

plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title("Confusion Matrix ANN")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig(r'Saved_Models/Images/confusion_matrix.png')
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred,digits=5))
