"""
A recurrent neural network (RNN) is a class of artificial 
neural networks where connections between nodes form a directed 
graph along a temporal sequence.

"""
#Part1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import indices # Visualise information
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Importing the training set
dataset_train = pd.read_csv("./Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values
print("data set... ")
print(dataset_train)
print("Original training set...  ")
print(training_set)

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
print("Scaling data from training set...")
print(training_set_scaled)

#Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train)
print(y_train)

#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train)

#Part2 - Building the RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

#Initialising the RNN
regressor = Sequential()
#Adding the first LST layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
#Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
#Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
#Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units=1))

#Compiling the RNN
regressor.compile(optimizer="adam", loss="mean_squared_error")

#Fitting the RNN to the Training Set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#Getting the real stock price of 2017
dataset_test = pd.read_csv("./Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

#Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Part3 - Making the predictions and visualising the results
plt.plot(real_stock_price, color = "red", label="Real Google Stock Price")
plt.plot(predicted_stock_price, color = "blue", label="Predicted google stock price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.show()