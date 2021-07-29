import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

griddata = pd.read_csv('Data_for_UCI_named.csv')
griddata.head()

Xgrid = griddata.iloc[:, 0:12]  # note that the Column 13 has the answer!
Xgrid.head()

ygrid = griddata.iloc[:, 13]
# 0 if unstable and 1 if stable
ygrid = [0 if x == 'unstable' else 1 for x in ygrid]

## Question 5.3.1
Xgridfulldata = np.array(Xgrid.values)
scaler = StandardScaler()
scaler.fit(Xgridfulldata)
Xgridscaled = scaler.transform(Xgridfulldata)

