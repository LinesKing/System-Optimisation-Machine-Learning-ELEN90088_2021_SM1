import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

dataw = pd.read_csv('wifi_localization.csv', names=[f"s{i}" for i in range(1, 8)] + ['Room Number'])
dataw.head()  # comment one to see the other
dataw.tail()

print(dataw.size, dataw.shape)

SRI = dataw.iloc[:, :7]
# a.shape
loc = dataw.iloc[:, 7] - 1
# loc.shape

# split into training and test sets
SRItrain, SRItest, loctrain, loctest = train_test_split(SRI, loc, random_state=3698557)

## Problem 3.1.2
# Set a random seed as you did in optimisation workshop by uncommenting the line below!
np.random.seed(3698557)

# PCA estimator
pca2 = PCA(n_components=2)
pca2.fit(SRI)
pca3 = PCA(n_components=3)
pca3.fit(SRI)
pca4 = PCA(n_components=4)
pca4.fit(SRI)

# Find singular values, variance ratios and apply dimensionality reduction
SV2 = pca2.singular_values_
SV3 = pca3.singular_values_
SV4 = pca4.singular_values_
VR2 = pca2.explained_variance_ratio_
VR3 = pca3.explained_variance_ratio_
VR4 = pca4.explained_variance_ratio_
X_pca2 = pca2.transform(SRI)
X_pca3 = pca3.transform(SRI)
X_pca4 = pca4.transform(SRI)
varsum = np.cumsum(pca4.explained_variance_ratio_)

# Plot data
plt.figure()
plt.scatter(X_pca2[:, 0], X_pca2[:, 1])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2])
plt.show()


