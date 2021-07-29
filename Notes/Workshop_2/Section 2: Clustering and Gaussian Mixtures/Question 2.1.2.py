from sklearn import cluster, datasets, mixture
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn import decomposition
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import random

## Problem 2.1.2
# Set a random seed as you did in optimisation workshop by uncommenting the line below!
np.random.seed(3698557)

noisy_moons = datasets.make_moons(n_samples=200, noise=0.05)
X = noisy_moons[0]  # data points
y = noisy_moons[1]  # 0, 1 labels of class, 50 each - giving us the ground truth

order_ind = np.argsort(y)  # order labels, 50 each class
X1 = X[order_ind[0:100]]  # class 1
X2 = X[order_ind[101:200]]  # class 2

# Use k-means clustering algorithm to divide the two moon data given above (X) into two clusters
# Change starting points (init='random')
randomlist = random.sample(range(0, 100), 4)
for k in range(4):
    kmeans = KMeans(init='random', n_clusters=2, random_state=randomlist[k]).fit(X)
    cluster_centres_X1 = kmeans.cluster_centers_[0]
    cluster_centres_X2 = kmeans.cluster_centers_[1]

    y = kmeans.labels_  # 0, 1 labels of class, 50 each
    order_ind = np.argsort(y)  # order labels, 50 each class
    X1 = X[order_ind[0:100]]  # class 1
    X2 = X[order_ind[101:200]]  # class 2

    # Plot data
    plt.figure()
    plt.title('random_state = %d' % randomlist[k])
    plt.scatter(X1[:, 0], X1[:, 1], color='blue')
    plt.scatter(X2[:, 0], X2[:, 1], color='red')
    plt.scatter(cluster_centres_X1[0], cluster_centres_X1[1], color='blue', marker='x', s=100, edgecolor='black',
                linewidth=3)
    plt.scatter(cluster_centres_X2[0], cluster_centres_X2[1], color='red', marker='x', s=100, edgecolor='black',
                linewidth=3)
    plt.show()


# Change number of clusters
# n = 3
kmeans = KMeans(n_clusters=3).fit(X)
cluster_centres_X1 = kmeans.cluster_centers_[0]
cluster_centres_X2 = kmeans.cluster_centers_[1]
cluster_centres_X3 = kmeans.cluster_centers_[2]

y = kmeans.labels_  # 0, 1 labels of class, 50 each
index_X1 = np.where(y == 0)
index_X2 = np.where(y == 1)
index_X3 = np.where(y == 2)
X1 = X[index_X1]  # class 1
X2 = X[index_X2]  # class 2
X3 = X[index_X3]  # class 3

# Plot data
plt.figure()
plt.scatter(X1[:, 0], X1[:, 1], color='blue')
plt.scatter(X2[:, 0], X2[:, 1], color='red')
plt.scatter(X3[:, 0], X3[:, 1], color='lime')
plt.scatter(cluster_centres_X1[0], cluster_centres_X1[1], color='blue', marker='X', s=100, edgecolor='black',
            linewidth=1)
plt.scatter(cluster_centres_X2[0], cluster_centres_X2[1], color='red', marker='X', s=100, edgecolor='black',
            linewidth=1)
plt.scatter(cluster_centres_X3[0], cluster_centres_X3[1], color='lime', marker='X', s=100, edgecolor='black',
            linewidth=1)
plt.show()


# n = 4
kmeans = KMeans(n_clusters=4).fit(X)
cluster_centres_X1 = kmeans.cluster_centers_[0]
cluster_centres_X2 = kmeans.cluster_centers_[1]
cluster_centres_X3 = kmeans.cluster_centers_[2]
cluster_centres_X4 = kmeans.cluster_centers_[3]

y = kmeans.labels_  # 0, 1 labels of class, 50 each
index_X1 = np.where(y == 0)
index_X2 = np.where(y == 1)
index_X3 = np.where(y == 2)
index_X4 = np.where(y == 3)
X1 = X[index_X1]  # class 1
X2 = X[index_X2]  # class 2
X3 = X[index_X3]  # class 3
X4 = X[index_X4]  # class 4

# Plot data
plt.figure()
plt.scatter(X1[:, 0], X1[:, 1], color='blue')
plt.scatter(X2[:, 0], X2[:, 1], color='red')
plt.scatter(X3[:, 0], X3[:, 1], color='lime')
plt.scatter(X4[:, 0], X4[:, 1], color='violet')
plt.scatter(cluster_centres_X1[0], cluster_centres_X1[1], color='blue', marker='X', s=100, edgecolor='black',
            linewidth=1)
plt.scatter(cluster_centres_X2[0], cluster_centres_X2[1], color='red', marker='X', s=100, edgecolor='black',
            linewidth=1)
plt.scatter(cluster_centres_X3[0], cluster_centres_X3[1], color='lime', marker='X', s=100, edgecolor='black',
            linewidth=1)
plt.scatter(cluster_centres_X4[0], cluster_centres_X4[1], color='violet', marker='X', s=100, edgecolor='black',
            linewidth=1)
plt.show()


# n = 5
kmeans = KMeans(n_clusters=5).fit(X)
cluster_centres_X1 = kmeans.cluster_centers_[0]
cluster_centres_X2 = kmeans.cluster_centers_[1]
cluster_centres_X3 = kmeans.cluster_centers_[2]
cluster_centres_X4 = kmeans.cluster_centers_[3]
cluster_centres_X5 = kmeans.cluster_centers_[4]

y = kmeans.labels_  # 0, 1 labels of class, 50 each
index_X1 = np.where(y == 0)
index_X2 = np.where(y == 1)
index_X3 = np.where(y == 2)
index_X4 = np.where(y == 3)
index_X5 = np.where(y == 4)
X1 = X[index_X1]  # class 1
X2 = X[index_X2]  # class 2
X3 = X[index_X3]  # class 3
X4 = X[index_X4]  # class 4
X5 = X[index_X5]  # class 4

# Plot data
plt.figure()
plt.scatter(X1[:, 0], X1[:, 1], color='blue')
plt.scatter(X2[:, 0], X2[:, 1], color='red')
plt.scatter(X3[:, 0], X3[:, 1], color='lime')
plt.scatter(X4[:, 0], X4[:, 1], color='violet')
plt.scatter(X5[:, 0], X5[:, 1], color='peru')
plt.scatter(cluster_centres_X1[0], cluster_centres_X1[1], color='blue', marker='X', s=100, edgecolor='black',
            linewidth=1)
plt.scatter(cluster_centres_X2[0], cluster_centres_X2[1], color='red', marker='X', s=100, edgecolor='black',
            linewidth=1)
plt.scatter(cluster_centres_X3[0], cluster_centres_X3[1], color='lime', marker='X', s=100, edgecolor='black',
            linewidth=1)
plt.scatter(cluster_centres_X4[0], cluster_centres_X4[1], color='violet', marker='X', s=100, edgecolor='black',
            linewidth=1)
plt.scatter(cluster_centres_X5[0], cluster_centres_X5[1], color='peru', marker='X', s=100, edgecolor='black',
            linewidth=1)
plt.show()
