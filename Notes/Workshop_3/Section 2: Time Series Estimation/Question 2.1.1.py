from math import sqrt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tools.eval_measures import mse

raw_data = pd.read_csv('two_houses.csv')
raw_data.head()

house1 = raw_data.iloc[1:, 2]
house2 = raw_data.iloc[1:, 3]

# plt.figure()
# plt.bar(np.arange(48 * 7), house1[0:48 * 7])
# plt.bar(np.arange(48 * 7), house2[0:48 * 7])
# plt.title('Half-Hourly Energy Consumption over One Week')
# plt.xlabel('Number of 30mins over One Week')
# plt.ylabel('KWh (per 30min)')
# plt.legend(['house1', 'house2'])
# plt.show()

### Problem 2.1.1
## house 1
# split into training and test sets
trainSize = 960
testSize = 48
train1, test1 = house1[0:trainSize], house1[trainSize:trainSize+testSize]

# walk-forward validation
# fit model
# model = ARIMA(train1, order=(2, 0, 2))
# model_fit = model.fit()
# predictions1 = model_fit.predict(start=trainSize, end=trainSize+testSize-1)

history1 = [x for x in train1]
predictions1 = []
for t in test1.index.values:
    model = ARIMA(history1, order=(1, 0, 1))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions1.append(yhat)
    obs = test1[t]
    history1.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# Forecast
# fc, se, conf = model_fit.forecast(testSize, alpha=0.05)  # 95% conf

# summary of fit model
print(model_fit.summary())

# density plot of residuals
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot(kind='kde')
# pyplot.show()

# evaluate forecasts
predictions1 = np.stack(predictions1, axis=0)
test1 = test1.to_numpy().reshape(-1, 1)
MSE = mse(test1, predictions1)
print('Test MSE: %.3f' % MSE)

# plot forecasts against actual outcomes
pyplot.plot(test1, label='test')
pyplot.plot(predictions1, color='red', label='predict')
pyplot.legend()
pyplot.show()

# model_fit.plot_predict(961, 984, dynamic=False)
# plt.show()

# Make as pandas series
# fc_series = pd.Series(fc, index=test1.index)
# lower_series = pd.Series(conf[:, 0], index=test1.index)
# upper_series = pd.Series(conf[:, 1], index=test1.index)
#
# # Plot
# plt.figure(figsize=(12, 5), dpi=100)
# # plt.plot(train1, label='training')
# plt.plot(test1, label='actual')
# plt.plot(fc_series, label='forecast')
# plt.fill_between(lower_series.index, lower_series, upper_series,
#                  color='k', alpha=.15)
# plt.title('Forecast vs Actuals')
# plt.legend(loc='upper left', fontsize=8)
# plt.show()


# # line plot of residuals
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()
# pyplot.show()

# # summary stats of residuals
# print(residuals.describe())
