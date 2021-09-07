import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

print(tf.__version__)

#Part 1
#Data Preprocessing
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1]

print(X)
print(y)

#Data encoding //Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X[:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
print(X)

#Splitting the dataset into the training set adn Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Part2 Building the ANN
#Initializing the ANN
ann = tf.keras.models.Sequential()
#Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))


#Part3 - Training the ANN

#for binay output, always use binary_crossentropy
ann.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)