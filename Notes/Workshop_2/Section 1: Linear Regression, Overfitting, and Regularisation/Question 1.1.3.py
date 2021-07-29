from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Problem 1.1.3
diodeDataSet = pd.read_csv('diode_dataset.csv', names=['Vf', 'If'])
# Note that if you don't put names to csv or into the function as above,
# pandas ignores the first row in calculations!
diodeDataSet.head()  # use .name[] to call data set

# Full data in correct form for sklearn
Vfulldata = np.array(diodeDataSet.values[:, 0]).reshape(-1, 1)  # reshape needed for sklearn functions
Ifulldata = np.array(diodeDataSet.values[:, 1]).reshape(-1, 1)

# Split into training and test sets
Vtrain, Vtest, Itrain, Itest = train_test_split(Vfulldata, Ifulldata)

# Transforms an input data matrix into a new data matrix of a given degree
# Built using a list of (key, value) pairs, where the key is a string containing the name you want to give this step
# and value is an estimator object
model = make_pipeline(PolynomialFeatures(degree=2), linear_model.LinearRegression(fit_intercept=False))

# fit to an order-2 polynomial data
model = model.fit(Vtrain, Itrain)

# Display results
a = model.named_steps.linearregression.coef_[0, 0]
b = model.named_steps.linearregression.coef_[0, 1]
c = model.named_steps.linearregression.coef_[0, 2]
ItrainPredict = a + b * Vtrain + c * Vtrain ** 2
ItestPredict = a + b * Vtest + c * Vtest ** 2
mse_train = mean_squared_error(Itrain, ItrainPredict)
mse_test = mean_squared_error(Itest, ItestPredict)

# Plot the new linear I-V curve
V = Vfulldata
I = a + b * V + c * V ** 2
plt.figure()
plt.plot(V[:], np.maximum(I[:], 0)[:])
plt.plot(Vfulldata, Ifulldata, '.')
plt.xlabel('Voltage, V')
plt.ylabel('Current, I')
plt.title('Diode I-V')
plt.grid()
plt.show()
