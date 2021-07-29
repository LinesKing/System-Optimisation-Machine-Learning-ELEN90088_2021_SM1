import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

np.random.seed(3698553)
commdata = pd.read_csv('svm_bonus_data.csv')

# Full data in correct form for sklearn
features = np.array(commdata.values[:, 1:6]).reshape(-1, 5)  # reshape needed for sklearn functions
label = np.array(commdata.values[:, 0])  # reshape needed for sklearn functions

# Apply PCA
pca = PCA(n_components=3)
pca.fit(features)
featurespca = pca.transform(features)

# Normalise data
scaler = StandardScaler()
scaler.fit(featurespca)
featuresscaled = scaler.transform(featurespca)
labelscaled = (1 + label)/2

# Split into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(featuresscaled, labelscaled, random_state=3698557)

## Linear regression
reg = linear_model.LinearRegression().fit(Xtrain, ytrain)

# Coefficients
a = reg.intercept_
b = reg.coef_
# Prediction
ytrainpred = reg.predict(Xtrain)
ytestpred = reg.predict(Xtest)
# MSE
mseRegTrain = mean_squared_error(ytrain, ytrainpred)
mseRegTest = mean_squared_error(ytest, ytestpred)
# Cross validation
scoresKmeansTrain = cross_val_score(reg, Xtrain, ytrain)
scoresKmeansTest = cross_val_score(reg, Xtest, ytest)

# Use k-means clustering
kmeans = KMeans(n_clusters=2, random_state=3328555).fit(Xtrain)

## Kmeans centroids
kmeansCentres1 = kmeans.cluster_centers_[0]
kmeansCentres2 = kmeans.cluster_centers_[1]
# Prediction
ytrainpred = kmeans.labels_
ytestpred = kmeans.predict(Xtest)
# MSE
mseKmeansTrain = mean_squared_error(ytrain, ytrainpred)
mseKmeansTest = mean_squared_error(ytest, ytestpred)
# Cross validation
scoresKmeansTrain = cross_val_score(kmeans, Xtrain, ytrain)
scoresKmeansTest = cross_val_score(kmeans, Xtest, ytest)
# Classification report
target_names = ['rogue agents', 'civilians']
print(classification_report(ytrain, ytrainpred, target_names=target_names))
print(classification_report(ytest, ytestpred, target_names=target_names))
# Plot
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(Xtrain[:, 0], Xtrain[:, 1], Xtrain[:, 2], c=ytrain, s=50, cmap='autumn')
ax.scatter3D(kmeansCentres1[0], kmeansCentres1[1], kmeansCentres1[2], s=50, marker='X')
ax.scatter3D(kmeansCentres2[0], kmeansCentres2[1], kmeansCentres2[2], s=50, marker='X')
plt.title("Kmeans")
plt.show()

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(Xtrain[:, 0], Xtrain[:, 1], Xtrain[:, 2], c=ytrainpred, s=50, cmap='autumn')
ax.scatter3D(kmeansCentres1[0], kmeansCentres1[1], kmeansCentres1[2], s=50, marker='X')
ax.scatter3D(kmeansCentres2[0], kmeansCentres2[1], kmeansCentres2[2], s=50, marker='X')
plt.title("Kmeans")
plt.show()

## Gaussian Mixture Models
gmm = GaussianMixture(n_components=2, covariance_type="full").fit(Xtrain)
# Prediction
ytrainpred = gmm.predict(Xtrain)
ytestpred = gmm.predict(Xtest)
# MSE
mseGmmTrain = mean_squared_error(ytrain, ytrainpred)
mseGmmTest = mean_squared_error(ytest, ytestpred)
# Cross validation
scoresGmmTrain = cross_val_score(gmm, Xtrain, ytrain)
scoresGmmTest = cross_val_score(gmm, Xtest, ytest)
# Classification report
target_names = ['rogue agents', 'civilians']
print(classification_report(ytrain, ytrainpred, target_names=target_names))
print(classification_report(ytest, ytestpred, target_names=target_names))


## SVM
svc = SVC(kernel='linear', C=10).fit(Xtrain, ytrain)

# Prediction
ytrainpred = svc.predict(Xtrain)
ytestpred = svc.predict(Xtest)
# MSE
mseSvmTrain = mean_squared_error(ytrain, ytrainpred)
mseSvmTest = mean_squared_error(ytest, ytestpred)
# Cross validation
scoresSvmTrain = cross_val_score(svc, Xtrain, ytrain)
scoresSvmTest = cross_val_score(svc, Xtest, ytest)
# Classification report
target_names = ['rogue agents', 'civilians']
print(classification_report(ytrain, ytrainpred, target_names=target_names))
print(classification_report(ytest, ytestpred, target_names=target_names))


## Logistic regression
lg = LogisticRegression(random_state=3698557).fit(Xtrain, ytrain)

# Coefficients
coefficients = lg.coef_
intercept = lg.intercept_
# Prediction
ytrainpred = lg.predict(Xtrain)
ytestpred = lg.predict(Xtest)
# MSE
mseLgTrain = mean_squared_error(ytrain, ytrainpred)
mseLgTest = mean_squared_error(ytest, ytestpred)
# Cross validation
scoresLgTrain = cross_val_score(lg, Xtrain, ytrain)
scoresLgTest = cross_val_score(lg, Xtest, ytest)
# Classification report
target_names = ['rogue agents', 'civilians']
print(classification_report(ytrain, ytrainpred, target_names=target_names))
print(classification_report(ytest, ytestpred, target_names=target_names))
