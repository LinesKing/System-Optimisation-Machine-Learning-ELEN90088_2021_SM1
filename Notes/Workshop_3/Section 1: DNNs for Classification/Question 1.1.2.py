import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn import cluster, datasets, mixture
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import datetime
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# from tensorflow import keras


# Set a random seed as you did in optimisation workshop by uncommenting the line below!
np.random.seed(5636994)

# Create a new moons data set
new_moons = datasets.make_moons(n_samples=400, noise=0.25)
Xm = new_moons[0]  # data points
ym = new_moons[1]  # 0, 1 labels of class, 200 each - giving us the ground truth

# Visualise the data set
# order_ind = np.argsort(ym)  # order labels, 200 each class
# Xm1 = Xm[order_ind[0:200]]  # class 1 - only for visualisation
# Xm2 = Xm[order_ind[201:400]]  # class 2 - only for visualisation
# plt.figure()
# plt.scatter(Xm1[:, 0], Xm1[:, 1], color='black')
# plt.scatter(Xm2[:, 0], Xm2[:, 1], color='red')
# plt.show()

# split into training and test sets
Xmtrain, Xmtest, ymtrain, ymtest = train_test_split(Xm, ym)

### Problem 1.1.2
## Use only one hidden layer or many more layers
# Define the DNN sequential model

model = Sequential()
model.add(tf.keras.Input(shape=(2,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

# log results
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model, iterating on the data in batches, record history
train_hist = model.fit(Xmtrain, ymtrain, epochs=1000, batch_size=400, verbose=1, callbacks=[tensorboard_callback])

print(model.summary())
weights = model.get_weights()  # Getting params
print(weights)

# Print the the actual score we have chosen and visualise the evolution of loss and accuracy over training epochs
score = model.evaluate(Xmtest, ymtest, batch_size=16, verbose=2)
print(score)

#train_hist.history

plt.figure()
plt.plot(train_hist.history['loss'])
plt.plot(train_hist.history['binary_accuracy'])
plt.xlabel('Epoch number')
plt.title('Training Loss and Accuracy')
plt.legend(['Loss', 'Accuracy'], loc='center right')
plt.show()

# Classification report
ytrainpred = np.ndarray.round(model.predict(Xmtrain))
ytestpred = np.ndarray.round(model.predict(Xmtest))
target_names = ['0', '1']
print(classification_report(ymtrain, ytrainpred, target_names=target_names))
print(classification_report(ymtest, ytestpred, target_names=target_names))
