import sklearn
from sklearn import cluster, datasets, mixture
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn import decomposition
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import LogNorm

## Problem 2.2.3
# Set a random seed as you did in optimisation workshop by uncommenting the line below!
np.random.seed(3698557)

noisy_moons = datasets.make_moons(n_samples=200, noise=0.05)
X_train = noisy_moons[0]  # data points
y_train = noisy_moons[1]  # 0, 1 labels of class, 50 each - giving us the ground truth

order_ind = np.argsort(y_train)  # order labels, 50 each class
X1 = X_train[order_ind[0:100]]  # class 1
X2 = X_train[order_ind[101:200]]  # class 2

# AIC & BIC for different n
IC = []  # Information criterion
NoC = np.arange(2, 50, 1)  # a metric to choose the number of components

for i in NoC:
    gmm = GaussianMixture(n_components=i + 2).fit(X_train)
    IC.append(gmm.aic(X_train) + gmm.bic(X_train))

min_IC = IC.index(min(IC))
n = NoC[min_IC]

gmmn = GaussianMixture(n_components=n, covariance_type='full', random_state=0).fit(X_train)
X_new = gmmn.sample(n_samples=200)

# Labels
y = X_new[1]
XX = X_new[0]

index_X1 = np.where(y == 0)
index_X2 = np.where(y == 1)
index_X3 = np.where(y == 2)
index_X4 = np.where(y == 3)
index_X5 = np.where(y == 4)
index_X6 = np.where(y == 5)

X1 = XX[index_X1[0]]  # class 1
X2 = XX[index_X2[0]]  # class 2
X3 = XX[index_X3[0]]  # class 3
X4 = XX[index_X4[0]]  # class 4
X5 = XX[index_X5[0]]  # class 5
X6 = XX[index_X6[0]]  # class 5

# display predicted scores by the model as a contour plot
x = np.linspace(-2.5, 2.5)
y = np.linspace(-2.5, 2.5)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmmn.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, levels=np.logspace(0, 2, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=10)
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=10)
plt.scatter(X3[:, 0], X3[:, 1], color='lime', s=10)
plt.scatter(X4[:, 0], X4[:, 1], color='violet', s=10)
plt.scatter(X5[:, 0], X5[:, 1], color='peru', s=10)
plt.scatter(X6[:, 0], X6[:, 1], color='black', s=10)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
