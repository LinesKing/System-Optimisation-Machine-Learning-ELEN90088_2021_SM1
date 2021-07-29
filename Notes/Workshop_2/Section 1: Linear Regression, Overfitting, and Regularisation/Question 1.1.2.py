from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## Problem 1.1.2
diodeDataSet = pd.read_csv('diode_dataset.csv', names=['Vf', 'If'])
# Note that if you don't put names to csv or into the function as above,
# pandas ignores the first row in calculations!
diodeDataSet.head()  # use .name[] to call data set

# Full data in correct form for sklearn
Vfulldata = np.array(diodeDataSet.values[:, 0]).reshape(-1, 1)  # reshape needed for sklearn functions
Ifulldata = np.array(diodeDataSet.values[:, 1]).reshape(-1, 1)

# Split into training and test sets
Vtrain, Vtest, Itrain, Itest = train_test_split(Vfulldata, Ifulldata, random_state=3698557)

# Linear regression
reg = linear_model.LinearRegression().fit(Vtrain, Itrain)

# Display results
a = reg.intercept_
b = reg.coef_
ItrainPredict = a + b * Vtrain
ItestPredict = a + b * Vtest
mse_train = mean_squared_error(Itrain, ItrainPredict)
mse_test = mean_squared_error(Itest, ItestPredict)

# Plot the new linear I-V curve
V = Vfulldata
I = a + b * V
plt.figure()
plt.plot(V[:], np.maximum(I[:], 0)[:])
plt.plot(Vfulldata, Ifulldata, '.')
plt.xlabel('Voltage, V')
plt.ylabel('Current, I')
plt.title('Diode I-V')
plt.grid()
plt.show()


