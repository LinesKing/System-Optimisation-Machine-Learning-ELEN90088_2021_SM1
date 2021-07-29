import numpy as np
import pandas as pd
import regressors
from matplotlib import pyplot as plt
from regressors.stats import coef_pval
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1
    ], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none', edgecolors='black')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


griddata = pd.read_csv('Data_for_UCI_named.csv')
griddata.head()

Xgrid = griddata.iloc[:, 0:12]  # note that the Column 13 has the answer!
Xgrid.head()

ygrid = griddata.iloc[:, 13]
# 0 if unstable and 1 if stable
ygrid = [0 if x == 'unstable' else 1 for x in ygrid]

Xgridfulldata = np.array(Xgrid.values)
scaler = StandardScaler()
scaler.fit(Xgridfulldata)
Xgridscaled = scaler.transform(Xgridfulldata)

# Split into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(Xgridscaled, ygrid)

## Question 5.3.4
# SVM
model = SVC(kernel='rbf')
model.fit(Xtrain, ytrain)
ytrainpred = model.predict(Xtrain)
ytestpred = model.predict(Xtest)

# Classification report
train_target_names = ['unstable', 'stable']
test_target_names = ['unstable', 'stable']
print(classification_report(ytrain, ytrainpred, target_names=train_target_names))
print(classification_report(ytest, ytestpred, target_names=test_target_names))

