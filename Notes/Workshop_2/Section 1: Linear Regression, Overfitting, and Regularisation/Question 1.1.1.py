import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo

diodeDataSet = pd.read_csv('diode_dataset.csv', names=['Vf', 'If'])
# Note that if you don't put names to csv or into the function as above,
# pandas ignores the first row in calculations!
diodeDataSet.head()  # use .name[] to call data set

## Problem 1.1.1
# A I-V linear regression unconstrained optimisation program
model = pyo.AbstractModel()
model.name = 'Diode'

# Note variables
N = len(diodeDataSet)
model.a = pyo.Var()
model.b = pyo.Var()


# Define model and constrains
def Diode(model):
    return sum((diodeDataSet.If[j] - (model.a + model.b * diodeDataSet.Vf[j])) ** 2 for j in range(N))


model.obj = pyo.Objective(rule=Diode, sense=pyo.minimize)

# Create an instance of the problem
diode = model.create_instance()

# Define solver
opt = pyo.SolverFactory('ipopt')

# Show results
opt.solve(diode)


def disp_soln(instance):
    output = []
    for v in instance.component_data_objects(pyo.Var, active=True):
        output.append(pyo.value(v))
        print(v, pyo.value(v))
    print(instance.obj, pyo.value(instance.obj))
    output.append(pyo.value(instance.obj))
    return output


# Display results
results = disp_soln(diode)
a = results[0]
b = results[1]

# Plot the new linear I-V curve
V = diodeDataSet.Vf
I = a + b * V
plt.figure()
plt.plot(V[:], np.maximum(I[:], 0))
plt.plot(diodeDataSet.Vf, diodeDataSet.If, '.')
plt.xlabel('Voltage, V')
plt.ylabel('Current, I')
plt.title('Diode I-V')
plt.grid()
plt.show()
