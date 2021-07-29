import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
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

## Problem 3.1.1
# Set a random seed as you did in optimisation workshop by uncommenting the line below!
np.random.seed(3698557)

# Use k-means clustering algorithm to divide the two moon data given above (X) into two clusters

kmeans = KMeans(n_clusters=4).fit(SRItrain)
loctrainpred = kmeans.labels_
loctestpred = kmeans.fit_predict(SRItest)
MIBS_train = metrics.adjusted_mutual_info_score(loctrain, loctrainpred)
MIBS_test = metrics.adjusted_mutual_info_score(loctest, loctestpred)

