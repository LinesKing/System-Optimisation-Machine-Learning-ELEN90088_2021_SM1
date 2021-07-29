import numpy as np
import pandas as pd
from matplotlib import pyplot, pyplot as plt
from statsmodels.tools.eval_measures import mse
from statsmodels.tsa.arima_model import ARIMA

raw_data = pd.read_csv('two_houses.csv')
raw_data.head()

house1 = raw_data.iloc[1:, 2]
house2 = raw_data.iloc[1:, 3]

## House 1
# Split into training and test sets
trainSize = 960
testSize = 48
train1, test1 = house1[0:trainSize], house1[trainSize:trainSize+testSize]


# Define a function to model ARMA
def ARMA_model(train, test, p, d, q):
    index = test.index
    # Define model
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()

    # summary of fit model
    print(model_fit.summary())
    print(model_fit.fittedvalues)
    print(model_fit.params)

    # Forecast
    # fc, se, conf = model_fit.forecast(testSize, alpha=0.05)  # 95% conf
    history = [x for x in train]
    fc = []
    conf = []
    for t in test.index.values:
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        confhat = output[2]
        fc.append(yhat)
        conf.append(confhat)
        obs = test[t]
        history.append(obs)
        # print('predicted=%f, expected=%f' % (yhat, obs))

    lower_series = pd.Series(conf[:, 0], index=index)
    upper_series = pd.Series(conf[:, 1], index=index)
    fc = np.stack(fc, axis=0)
    conf = np.stack(conf, axis=0)


    test = test.to_numpy().reshape(-1, 1)
    MSE = mse(test, fc)
    print('Test MSE: %.3f' % MSE)

    # Plot
    # plot forecasts against actual outcomes
    plt.plot(test, label='test')
    plt.plot(fc, color='red', label='predict')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.15, label='95% confidence interval')
    plt.legend()
    plt.show()

## House 1

ARMA_model(train1, test1, 1, 0, 1)



# sliding window function for next 24 hourly estimate
# see, e.g. https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f
# or https://machinelearningmastery.com/reframe-time-series-forecasting-problem/
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


# get the estimate data for house1 and house2
X1, yday1 = house_data(house1[0:8736])
X2, yday2 = house_data(house2[0:8736])

# split into training and test sets for house 1
X1train, X1test, y1train, y1test = train_test_split(X1, yday1, test_size=0.01)
X1train = np.array(X1train).reshape(X1train.shape[0], X1train.shape[1], 1)
X1test = np.array(X1test).reshape(X1test.shape[0], X1test.shape[1], 1)


# univariate multi-step lstm


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


# train the model
def build_model(train_x, train_y):
    # define parameters
    verbose, epochs, batch_size = 1, 32, 20
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(LSTM(20, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


# evaluate a single model
def evaluate_model(train_x, train_y, test_x, test_y):
    # fit model
    model = build_model(train_x, train_y)
    # history is a list of weekly data
    history = [x for x in test_x]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test_x)):
        # predict the week
        yhat = model.predict(test_x, verbose=0)
        yhat_sequence = yhat[0]
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test_x[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test_y[:, :, 0], predictions)
    return score, scores



# evaluate model and get scores
n_input = 48
score, scores = evaluate_model(X1train, y1train, X1test, y1test)
# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
pyplot.plot(scores, marker='o', label='lstm')
pyplot.show()