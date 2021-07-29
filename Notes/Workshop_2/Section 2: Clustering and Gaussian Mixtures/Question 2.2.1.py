from sklearn import datasets
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

## Problem 2.2.1
# Set a random seed as you did in optimisation workshop by uncommenting the line below!
np.random.seed(3698557)

noisy_moons = datasets.make_moons(n_samples=200, noise=0.05)
X_train = noisy_moons[0]  # data points
y_train = noisy_moons[1]  # 0, 1 labels of class, 50 each - giving us the ground truth

order_ind = np.argsort(y_train)  # order labels, 50 each class
X1 = X_train[order_ind[0:100]]  # class 1
X2 = X_train[order_ind[101:200]]  # class 2

# Gaussian Mixture Models
gmm = GaussianMixture(n_components=2, covariance_type="full").fit(X_train)

# Probabilities and labels
p = gmm.predict_proba(X_train)
y = gmm.predict(X_train)

index_X1 = np.where(y == 0)
index_X2 = np.where(y == 1)

X1 = X_train[index_X1]  # class 1
X2 = X_train[index_X2]  # class 2

# display predicted scores by the model as a contour plot
x = np.linspace(-2.5, 2.5)
y = np.linspace(-2.5, 2.5)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=50.0),
                 levels=np.logspace(0, 2, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=10)
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=10)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()


