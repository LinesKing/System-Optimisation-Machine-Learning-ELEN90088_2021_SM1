import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

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

## Problem 5.1.2
# Logistic
lg = LogisticRegression(random_state=3698557).fit(X_toy, y_toy)
ytoy_testPred = 1/(1+np.exp(-(lg.intercept_ + lg.coef_ * Xtoy_test[:, np.newaxis])))
# ytoy_testPred = lg.predict(Xtoy_test[:, np.newaxis])

# visualise
plt.figure()
plt.plot(Xtoy_test, ytoy_testPred)
plt.scatter(X_toy, y_toy)
plt.show()

# Linear
reg = linear_model.LinearRegression().fit(X_toy, y_toy)
ytoy_testPred = reg.predict(Xtoy_test[:, np.newaxis])

# visualise
plt.figure()
plt.plot(Xtoy_test, ytoy_testPred)
plt.scatter(X_toy, y_toy)
plt.show()
