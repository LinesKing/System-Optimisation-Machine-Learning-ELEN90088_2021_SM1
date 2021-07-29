import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot, pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf

raw_data = pd.read_csv('two_houses.csv')
raw_data.head()

house1 = raw_data.iloc[1:, 2]
house2 = raw_data.iloc[1:, 3]


def house_data(inseries):
    window_size = 48 + 48
    series = inseries
    series_s = inseries.copy()
    for i in range(window_size):
        series = pd.concat([series, series_s.shift(-(i + 1))], axis=1)
    series.dropna(axis=0, inplace=True)
    X = series.iloc[:, 0:48]
    yday = series.iloc[:, 48:48 + 48]  # next day
    return X, yday


# train the model
def build_model(Xtrain, ytrain):
    # prepare data
    # define parameters
    verbose, epochs, batch_size = 1, 128, 20
    n_timesteps, n_features, n_outputs = Xtrain.shape[1], Xtrain.shape[2], ytrain.shape[1]
    # define model
    model = Sequential()
    model.add(LSTM(20, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model_fit = model.fit(Xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model, model_fit


# get the estimate data for house1 and house2
X1, yday1 = house_data(house1[0:8736])
X2, yday2 = house_data(house2[0:8736])

# split into training and test sets for house 1
X1train, X1test, y1train, y1test = train_test_split(X1, yday1)
X1train = np.array(X1train).reshape(X1train.shape[0], X1train.shape[1], 1)
X1test = np.array(X1test).reshape(X1test.shape[0], X1test.shape[1], 1)

# fit model
model, model_fit = build_model(X1train, y1train)
y1pred = model.predict(X1test)

plt.figure()
plt.plot(model_fit.history['loss'])
plt.xlabel('Epoch number')
plt.title('Training Loss')
plt.show()

# plot forecasts against actual outcomes
pyplot.plot(y1test.to_numpy().reshape(-1, 1), label='test')
pyplot.plot(y1pred.reshape(-1, 1), color='red', label='predict')
pyplot.legend()
pyplot.show()

MSE = mean_squared_error(y1test.to_numpy().reshape(-1, 1), y1pred.reshape(-1, 1))
