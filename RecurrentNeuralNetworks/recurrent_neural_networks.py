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
X_train = np.reshape(X_train, (-1, 1, 1))
print(X_train)

#Part2 - Building the RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

#Initialising the RNN
regressor = Sequential()
#Adding the first LST layer and some Dropout regularisation
#regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (np.array(X_train)[indices.astype(int)], 1)))

#Part3 - Making the predictions and visualising the results
