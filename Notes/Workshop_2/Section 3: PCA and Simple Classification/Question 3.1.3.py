import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
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
SRItrain, SRItest, loctrain, loctest = train_test_split(SRI, loc)

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

## Problem 3.1.3
# 2 features
SRItrain, SRItest, loctrain, loctest = train_test_split(X_pca2, loc)
kmeans = KMeans(n_clusters=4).fit(SRItrain)
loctrainpred = kmeans.labels_
loctestpred = kmeans.fit_predict(SRItest)
MIBS_train2 = metrics.adjusted_mutual_info_score(loctrain, loctrainpred)
MIBS_test2 = metrics.adjusted_mutual_info_score(loctest, loctestpred)

# 3 features
SRItrain, SRItest, loctrain, loctest = train_test_split(X_pca3, loc)
kmeans = KMeans(n_clusters=4).fit(SRItrain)
loctrainpred = kmeans.labels_
loctestpred = kmeans.fit_predict(SRItest)
MIBS_train3 = metrics.adjusted_mutual_info_score(loctrain, loctrainpred)
MIBS_test3 = metrics.adjusted_mutual_info_score(loctest, loctestpred)

# 4 features
SRItrain, SRItest, loctrain, loctest = train_test_split(X_pca4, loc)
kmeans = KMeans(n_clusters=4).fit(SRItrain)
loctrainpred = kmeans.labels_
loctestpred = kmeans.fit_predict(SRItest)
MIBS_train4 = metrics.adjusted_mutual_info_score(loctrain, loctrainpred)
MIBS_test4 = metrics.adjusted_mutual_info_score(loctest, loctestpred)
