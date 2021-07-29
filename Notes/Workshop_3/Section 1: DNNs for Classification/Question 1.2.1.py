import pandas as pd
import numpy as np
from matplotlib import pyplot
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
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from collections import deque

dataw = pd.read_csv('wifi_localization.csv', names=[f"s{i}" for i in range(1, 8)] + ['Room Number'])
dataw.head()  # comment one to see the other
dataw.tail()

print(dataw.size, dataw.shape)

SRI = dataw.iloc[:, :7]
# a.shape
loc = dataw.iloc[:, 7] - 1

# loc.shape
loc = loc.to_numpy()
locBin = np.array([list(np.binary_repr(x, 2)) for x in loc], dtype=int)

# loc1Hot = np.zeros((loc.size, loc.max() + 1))
# loc1Hot[np.arange(loc.size), loc] = 1

# split into training and test sets
SRItrain, SRItest, loctrain, loctest = train_test_split(SRI, locBin)

# Set a random seed as you did in optimisation workshop by uncommenting the line below!
np.random.seed(5636994)

### Problem 1.2.1
## Use only one hidden layer or many more layers
# Define the DNN sequential model

model = Sequential()
model.add(tf.keras.Input(shape=(7,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

# Train the model, iterating on the data in batches, record history
train_hist = model.fit(SRItrain, loctrain, epochs=100, batch_size=16, verbose=0)

# Print the the actual score we have chosen and visualise the evolution of loss and accuracy over training epochs
score = model.evaluate(SRItest, loctest, batch_size=16, verbose=0)
print(score)

# train_hist.history

plt.figure()
plt.plot(train_hist.history['loss'])
plt.plot(train_hist.history['binary_accuracy'])
plt.xlabel('Epoch number')
plt.title('Training Loss and Accuracy')
plt.legend(['Loss', 'Accuracy'], loc='center right')
plt.show()

# Classification report
loctrainpred = np.ndarray.round(model.predict(SRItrain))
loctestpred = np.ndarray.round(model.predict(SRItest))

# calculate scores
b2d = 1 << np.arange(loctrain.shape[-1] - 1, -1, -1)
loctrain = np.rint(loctrain.dot(b2d)).reshape(-1, 1)
loctrainpred = np.rint(loctrainpred.dot(b2d)).reshape(-1, 1)
loctest = np.rint(loctest.dot(b2d)).reshape(-1, 1)
loctestpred = np.rint(loctestpred.dot(b2d)).reshape(-1, 1)


# loctrain = [np.where(r == 1)[0][0] for r in loctrain]
# loctrainpred = [np.where(r == 1)[0][0] for r in loctrainpred]
# loctest = [np.where(r == 1)[0][0] for r in loctest]
# loctestpred = [np.where(r == 1)[0][0] for r in loctestpred]

def d2ih(loc):
    loc1Hot = np.zeros((loc.size, loc.max() + 1))
    for i in np.arange(loc.size):
        loc1Hot[i, loc[i]] = 1
    return loc1Hot


loctrain = d2ih(loctrain.astype(int))
loctrainpred = d2ih(loctrainpred.astype(int))
loctest = d2ih(loctest.astype(int))
loctestpred = d2ih(loctestpred.astype(int))

train_auc = roc_auc_score(loctrain, loctrainpred, multi_class='ovr')
test_auc = roc_auc_score(loctest, loctestpred, multi_class='ovr')
# summarize scores
print('Train: ROC AUC=%.3f' % train_auc)
print('Test: ROC AUC=%.3f' % test_auc)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(loctest[:, i], loctestpred [:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(4):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
