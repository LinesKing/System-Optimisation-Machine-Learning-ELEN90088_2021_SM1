import numpy as np
import pandas as pd
import regressors
from regressors.stats import coef_pval
from sklearn.linear_model import LogisticRegression
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

# Split into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(Xgridscaled, ygrid)

## Question 5.3.3
# Logistic
lg = LogisticRegression(random_state=3698557).fit(Xtrain, ytrain)

coefficient = lg.coef_
intercept = lg.intercept_

# P value
p = coef_pval(lg, Xtest, ytest)
