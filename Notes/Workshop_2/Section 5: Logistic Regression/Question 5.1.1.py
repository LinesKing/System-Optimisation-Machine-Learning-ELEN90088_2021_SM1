import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

# Generate a toy dataset
toy_samples = 50
X_toy = np.linspace(-5, 5, toy_samples)
Xtoy_test = np.linspace(-5, 5, 200)
# gaussian noise added
X_toy = X_toy + 2 * np.random.normal(size=toy_samples)
# upper, lower bound
X_toy = np.clip(X_toy, -5, 5).reshape(-1, 1)
# create labels
y_toy = ((np.sign(X_toy) + 1) / 2.0).ravel()

# # visualise
# plt.figure()
# plt.scatter(X_toy, y_toy)
# plt.show()

## Problem 5.1.1
# Logistic
lg = LogisticRegression(random_state=3698557).fit(X_toy, y_toy)
# y_toyPred = lg.predict(X_toy)
# y_toyPred = 1/(1+np.exp(-(lg.intercept_ + lg.coef_ * X_toy)))
y_toyPred = lg.predict(X_toy)
mse_lg = mean_squared_error(y_toy, y_toyPred)

# Linear
# reg = linear_model.LinearRegression().fit(X_toy, y_toy)
# y_toyPred = reg.predict(X_toy)
# mse_reg = mean_squared_error(y_toy, y_toyPred)

# visualise
plt.figure()
plt.scatter(X_toy, y_toy)
plt.show()
