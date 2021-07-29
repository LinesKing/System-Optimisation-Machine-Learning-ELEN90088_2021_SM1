import numpy as np
import pandas as pd
from IPython.core.display import Markdown, display
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

## Question 5.3.2
# Split into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(Xgridscaled, ygrid)

# Logistic
C = np.logspace(-2, 2, 5)
fig, ax = plt.subplots()
for c in C:
    display(Markdown(r"C = %f" % (c)))
    lg = LogisticRegression(random_state=3698557, C=c).fit(Xtrain, ytrain)
    # ytrain = np.array(ytrain).T
    # ytest = np.array(ytest).T
    # ytrainPred = 1/(1+np.exp(-(lg.intercept_ + lg.coef_.dot(Xtrain.T))))
    # ytestPred = 1/(1+np.exp(-(lg.intercept_ + lg.coef_ .dot(Xtest.T))))
    ytrainPred = lg.predict(Xtrain)
    ytestPred = lg.predict(Xtest)

    # Classification report
    train_target_names = ['unstable', 'stable']
    test_target_names = ['unstable', 'stable']
    print('train:')
    print(classification_report(ytrain, np.squeeze(ytrainPred).T, target_names=train_target_names))
    print('test:')
    print(classification_report(ytest, np.squeeze(ytestPred).T, target_names=test_target_names))

    # Plot ROC curve
    metrics.plot_roc_curve(lg, Xtest, ytest, ax=ax)

plt.show()
