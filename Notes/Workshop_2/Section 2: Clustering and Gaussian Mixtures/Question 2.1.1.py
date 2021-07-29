from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

## Problem 2.1.1
# Set a random seed as you did in optimisation workshop by uncommenting the line below!
np.random.seed(3698557)

noisy_moons = datasets.make_moons(n_samples=200, noise=0.05)
X = noisy_moons[0]  # data points
y = noisy_moons[1]  # 0, 1 labels of class, 50 each - giving us the ground truth

order_ind = np.argsort(y)  # order labels, 50 each class
X1 = X[order_ind[0:100]]  # class 1
X2 = X[order_ind[101:200]]  # class 2

# Use k-means clustering algorithm to divide the two moon data given above (X) into two clusters
kmeans = KMeans(n_clusters=2).fit(X)
cluster_centres_X1 = kmeans.cluster_centers_[0]
cluster_centres_X2 = kmeans.cluster_centers_[1]

y = kmeans.labels_  # 0, 1 labels of class, 50 each
order_ind = np.argsort(y)  # order labels, 50 each class
X1 = X[order_ind[0:100]]  # class 1
X2 = X[order_ind[101:200]]  # class 2

# Plot data
plt.figure()
plt.scatter(X1[:, 0], X1[:, 1], color='blue')
plt.scatter(X2[:, 0], X2[:, 1], color='red')
plt.scatter(cluster_centres_X1[0], cluster_centres_X1[1], color='blue', marker='x', s=100, edgecolor='black',
            linewidth=3)
plt.scatter(cluster_centres_X2[0], cluster_centres_X2[1], color='red', marker='x', s=100, edgecolor='black',
            linewidth=3)
plt.show()
