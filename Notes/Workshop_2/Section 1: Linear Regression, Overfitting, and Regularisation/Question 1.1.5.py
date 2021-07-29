from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Problem 1.1.5
diodeDataSet = pd.read_csv('diode_dataset.csv', names=['Vf', 'If'])
# Note that if you don't put names to csv or into the function as above,
# pandas ignores the first row in calculations!
diodeDataSet.head()  # use .name[] to call data set

# Full data in correct form for sklearn
Vfulldata = np.array(diodeDataSet.values[:, 0]).reshape(-1, 1)  # reshape needed for sklearn functions
Ifulldata = np.array(diodeDataSet.values[:, 1]).reshape(-1, 1)

# Split into training and test sets
Vtrain, Vtest, Itrain, Itest = train_test_split(Vfulldata, Ifulldata, test_size=0.4, random_state=3698557)

## Order = 4
# Transforms an input data matrix into a new data matrix of a given degree
# Built using a list of (key, value) pairs, where the key is a string containing the name you want to give this step
# and value is an estimator object
model = make_pipeline(PolynomialFeatures(degree=4), linear_model.LinearRegression(fit_intercept=False))

# Fit to an order-4 polynomial data
model = model.fit(Vtrain, Itrain)

# Cross-validation score
# score = model.score(Vtest, Itest)
scores = cross_val_score(model, Vtest, Itest)

# Display results
a = model.named_steps.linearregression.coef_[0, 0]
b = model.named_steps.linearregression.coef_[0, 1]
c = model.named_steps.linearregression.coef_[0, 2]
d = model.named_steps.linearregression.coef_[0, 3]
e = model.named_steps.linearregression.coef_[0, 4]
IPredict = a + b * Vfulldata + c * Vfulldata ** 2 + d * Vfulldata ** 3 + e * Vfulldata ** 4
mse = mean_squared_error(Ifulldata, IPredict)

# Plot the new linear I-V curve
V = Vfulldata
I = a + b * V + c * V ** 2 + d * V ** 3 + e * V ** 4
plt.figure()
plt.plot(V[:], np.maximum(I[:], 0)[:])
plt.plot(Vfulldata, Ifulldata, '.')
plt.xlabel('Voltage, V')
plt.ylabel('Current, I')
plt.title('Diode I-V')
plt.grid()
plt.show()

## Order = 6
# Transforms an input data matrix into a new data matrix of a given degree
# Built using a list of (key, value) pairs, where the key is a string containing the name you want to give this step
# and value is an estimator object
model = make_pipeline(PolynomialFeatures(degree=6), linear_model.LinearRegression(fit_intercept=False))

# Fit to an order-6 polynomial data
model = model.fit(Vtrain, Itrain)

# Cross-validation score
# score = model.score(Vtest, Itest)
scores = cross_val_score(model, Vtrain, Itrain, cv=5)

# Display results
a = model.named_steps.linearregression.coef_[0, 0]
b = model.named_steps.linearregression.coef_[0, 1]
c = model.named_steps.linearregression.coef_[0, 2]
d = model.named_steps.linearregression.coef_[0, 3]
e = model.named_steps.linearregression.coef_[0, 4]
f = model.named_steps.linearregression.coef_[0, 5]
g = model.named_steps.linearregression.coef_[0, 6]
IPredict = a + b * Vfulldata + c * Vfulldata ** 2 + d * Vfulldata ** 3 + e * Vfulldata ** 4 \
           + f * Vfulldata ** 5 + g * Vfulldata ** 6
mse = mean_squared_error(Ifulldata, IPredict)

# Plot the new linear I-V curve
V = Vfulldata
I = a + b * V + c * V ** 2 + d * V ** 3 + e * V ** 4 + f * V ** 5 + g * V ** 6
plt.figure()
plt.plot(V[:], np.maximum(I[:], 0)[:])
plt.plot(Vfulldata, Ifulldata, '.')
plt.xlabel('Voltage, V')
plt.ylabel('Current, I')
plt.title('Diode I-V')
plt.grid()
plt.show()

