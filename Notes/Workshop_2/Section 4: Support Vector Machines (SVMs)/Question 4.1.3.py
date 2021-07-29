import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# this is not necessary if you run it in the cell earlier...
np.random.seed(3698557)

# Create a new moons data set
new_moons = datasets.make_moons(n_samples=400, noise=0.15)
Xm = new_moons[0]  # data points
ym = new_moons[1]  # 0, 1 labels of class, 200 each - giving us the ground truth

# Split into training and test sets
Xmtrain, Xmtest, ymtrain, ymtest = train_test_split(Xm, ym)

## Problem 4.1.3
# Find good  C anf gamma
tuned_parameters = [{'kernel': ['rbf'], 'gamma': np.logspace(-1, 0, num=10), 'C': np.logspace(1, 3, num=10)}]
gs = GridSearchCV(SVC(), param_grid=tuned_parameters, scoring={'AUC': 'roc_auc'}, refit='AUC', return_train_score=True)
gs.fit(Xmtrain, ymtrain)
results = gs.cv_results_

print("Best parameters set found on development set:")
print()
print(gs.best_params_)
print()
# print("Grid scores on development set:")
# print()
# aucs = gs.cv_results_['mean_train_AUC']
# for auc, params in zip(aucs, gs.cv_results_['params']):
#         print("%0.3f for %r" % (auc, params))


# Plot
tuned_parameters = [{'kernel': ['rbf'], 'gamma': np.logspace(-5, 0, num=100), 'C': [gs.best_params_['C']]}]
gs = GridSearchCV(SVC(), param_grid=tuned_parameters, scoring={'AUC': 'roc_auc'}, refit='AUC', return_train_score=True)
gs.fit(Xmtrain, ymtrain)
results = gs.cv_results_
aucs = gs.cv_results_['mean_train_AUC']

# Display grid
plt.grid(True, which="both")
plt.loglog(np.logspace(-5, 0, num=100), aucs)
plt.title('AUC vs gamma for the best C')
plt.xlabel('gamma')
plt.ylabel('AUC')
plt.show()
