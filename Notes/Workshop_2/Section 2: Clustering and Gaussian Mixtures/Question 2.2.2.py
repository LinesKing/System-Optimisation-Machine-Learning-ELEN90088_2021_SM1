from sklearn import datasets
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

## Problem 2.2.2
# Set a random seed as you did in optimisation workshop by uncommenting the line below!
np.random.seed(3698557)

noisy_moons = datasets.make_moons(n_samples=200, noise=0.05)
X_train = noisy_moons[0]  # data points
y_train = noisy_moons[1]  # 0, 1 labels of class, 50 each - giving us the ground truth

order_ind = np.argsort(y_train)  # order labels, 50 each class
X1 = X_train[order_ind[0:100]]  # class 1
X2 = X_train[order_ind[101:200]]  # class 2

# Gaussian Mixture Models(n=3)
gmm = GaussianMixture(n_components=3, covariance_type="full").fit(X_train)

# Probabilities and labels
p = gmm.predict_proba(X_train)
y = gmm.predict(X_train)

index_X1 = np.where(y == 0)
index_X2 = np.where(y == 1)
index_X3 = np.where(y == 2)

X1 = X_train[index_X1]  # class 1
X2 = X_train[index_X2]  # class 2
X3 = X_train[index_X3]  # class 3

# Check AIC and BIC
AIC3 = gmm.aic(X_train)
BIC3 = gmm.bic(X_train)

# display predicted scores by the model as a contour plot
x = np.linspace(-2.5, 2.5)
y = np.linspace(-2.5, 2.5)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, levels=np.logspace(0, 2, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=10)
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=10)
plt.scatter(X3[:, 0], X3[:, 1], color='lime', s=10)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()


# Gaussian Mixture Models(n=4)
gmm = GaussianMixture(n_components=4, covariance_type="full").fit(X_train)

# Probabilities and labels
p = gmm.predict_proba(X_train)
y = gmm.predict(X_train)

index_X1 = np.where(y == 0)
index_X2 = np.where(y == 1)
index_X3 = np.where(y == 2)
index_X4 = np.where(y == 3)

X1 = X_train[index_X1]  # class 1
X2 = X_train[index_X2]  # class 2
X3 = X_train[index_X3]  # class 3
X4 = X_train[index_X4]  # class 4

# Check AIC and BIC
AIC4 = gmm.aic(X_train)
BIC4 = gmm.bic(X_train)

# display predicted scores by the model as a contour plot
x = np.linspace(-2.5, 2.5)
y = np.linspace(-2.5, 2.5)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, levels=np.logspace(0, 2, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=10)
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=10)
plt.scatter(X3[:, 0], X3[:, 1], color='lime', s=10)
plt.scatter(X4[:, 0], X4[:, 1], color='violet', s=10)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()


# Gaussian Mixture Models(n=5)
gmm = GaussianMixture(n_components=5, covariance_type="full").fit(X_train)

# Probabilities and labels
p = gmm.predict_proba(X_train)
y = gmm.predict(X_train)

index_X1 = np.where(y == 0)
index_X2 = np.where(y == 1)
index_X3 = np.where(y == 2)
index_X4 = np.where(y == 3)
index_X5 = np.where(y == 4)

X1 = X_train[index_X1]  # class 1
X2 = X_train[index_X2]  # class 2
X3 = X_train[index_X3]  # class 3
X4 = X_train[index_X4]  # class 4
X5 = X_train[index_X5]  # class 5

# Check AIC and BIC
AIC5 = gmm.aic(X_train)
BIC5 = gmm.bic(X_train)

# display predicted scores by the model as a contour plot
x = np.linspace(-2.5, 2.5)
y = np.linspace(-2.5, 2.5)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, levels=np.logspace(0, 2, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=10)
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=10)
plt.scatter(X3[:, 0], X3[:, 1], color='lime', s=10)
plt.scatter(X4[:, 0], X4[:, 1], color='violet', s=10)
plt.scatter(X5[:, 0], X5[:, 1], color='peru', s=10)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()

# AIC & BIC for different n
AIC = []  # Akaike information criterion
BIC = []  # Bayesian information criterion
NoC = np.arange(2, 50, 1)  # a metric to choose the number of components

for i in NoC:
    gmm = GaussianMixture(n_components=i+2).fit(X_train)
    AIC.append(gmm.aic(X_train))
    BIC.append(gmm.bic(X_train))

min_AIC = AIC.index(min(AIC))
min_BIC = BIC.index(min(BIC))

plt.figure()
plt.plot(NoC, AIC)
plt.scatter(NoC[min_AIC], AIC[min_AIC], color='red', marker='X')
plt.plot(NoC, BIC)
plt.scatter(NoC[min_BIC], BIC[min_BIC], color='blue', marker='X')
plt.xlabel('n')
plt.ylabel('AIC/BIC')
plt.title('AIC/BIC')
plt.legend(['AIC', 'BIC'])
plt.grid()
plt.show()
