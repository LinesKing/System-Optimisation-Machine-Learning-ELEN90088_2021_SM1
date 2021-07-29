import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# helper function to visualise decision boundary, uses the svm model as input
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


# this is not necessary if you run it in the cell earlier...
np.random.seed(3698557)

# Create a new moons data set
new_moons = datasets.make_moons(n_samples=400, noise=0.15)
Xm = new_moons[0]  # data points
ym = new_moons[1]  # 0, 1 labels of class, 200 each - giving us the ground truth

## Problem 4.1.1
# Split into training and test sets
c = np.logspace(-2, 2, num=5)
Xmtrain, Xmtest, ymtrain, ymtest = train_test_split(Xm, ym)
for ci in c:
    model = SVC(kernel='linear', C=ci)
    model.fit(Xmtrain, ymtrain)
    ymtrainpred = model.predict(Xmtrain)
    ymtestpred = model.predict(Xmtest)

    # Classification report
    train_target_names = ['train class 1', 'train class 2']
    test_target_names = ['test class 1', 'test class 2']
    print(classification_report(ymtrain, ymtrainpred, target_names=train_target_names))
    print(classification_report(ymtest, ymtestpred, target_names=test_target_names))

    # Plot
    plt.scatter(Xmtrain[:, 0], Xmtrain[:, 1], c=ymtrain, s=50, cmap='autumn')
    plot_svc_decision_function(model)
    plt.title('c = %f' % ci)
    plt.show()
