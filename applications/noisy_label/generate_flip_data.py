import numpy as np
import pickle
import copy
import random
from applications.noisy_label.utils import *


# load train data
for i in range(10):
    with open('../arrays/raw_data/train/df_' + str(i) + '.npz', "rb") as f:
        x = np.load(f)["x"]
        y = np.load(f)["y"]
        if i == 0:
            X_train = x
            y_train = y
        else:
            X_train = np.concatenate((X_train, x), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)
print("Train data:", X_train.shape, y_train.shape)

# load test data
for i in range(10):
    with open('../arrays/raw_data/test/df_' + str(i) + '.npz', "rb") as f:
        x = np.load(f)["x"]
        y = np.load(f)["y"]
        if i == 0:
            X_test = x
            y_test = y
        else:
            X_test = np.concatenate((X_test, x), axis=0)
            y_test = np.concatenate((y_test, y), axis=0)
print("Test data:", X_test.shape, y_test.shape)

# select data
X_data = X_train
y_data = y_train
X_test_data = X_test
y_test_data = y_test

# create copy before modifying labels
y_data_orig = copy.deepcopy(y_data)

# benign data
X_benign = []
y_benign = []

# flipped data
X_flip = []
y_flip = []

# define number of train data points
x = 150000
flip = np.zeros(x)

# generate 10% flipped data
for i in range(x // 10):
    j = np.random.randint(0, x)
    while flip[j] == 1:
        j = np.random.randint(0, x)
    flip[j] = 1
    y_data[j] = 1 - y_data[j]
    X_flip.append(X_data[j])
    y_flip.append(y_data[j])
for i in range(x):
    if flip[i] == 0:
        X_benign.append(X_data[i])
        y_benign.append(y_data[i])

X_benign = np.arrays(X_benign)
y_benign = np.arrays(y_benign)
X_flip = np.arrays(X_flip)
y_flip = np.arrays(y_flip)

# store flipped data
for i in range(10):
    np.savez_compressed("../applications/noisy_label/flip_arrays/raw_data/train/" + "df_" + str(i) + ".npz", x=X_data[i * 15000:i * 15000 + 15000],
                        y=y_data[i * 15000:i * 15000 + 15000])
for i in range(10):
    np.savez_compressed("../applications/noisy_label/flip_arrays/raw_data/test/" + "df_" + str(i) + ".npz", x=X_test_data[i * 3750:i * 3750 + 3750],
                        y=y_test_data[i * 3750:i * 3750 + 3750])
print("Deep features for test data saved")


# compare performance of model trained on original and model trained on flipped data
# model with original data
model_1 = DenseNet121()
train_evaluate("train", model_1, X_data, y_data_orig, batch_size=8, epochs=3)
score, loss = train_evaluate("val", model_1, X_test_data, y_test_data, batch_size=8)
print("---Score on original data:", score)

# model with modified data
model_2 = DenseNet121()
train_evaluate("train", model_2, X_data, y_data, batch_size=8, epochs=3)
score, loss = train_evaluate("val", model_2, X_test_data, y_test_data, batch_size=8)
print("---Score on flipped data:", score)

# only choose 100000 data points due to complexity of KNN Shap calculations
y_train_old = y_data_orig[:100000]
y_train_new = y_data[:100000]


# create array that stores information about flipped data
flip = []
counter_same = 0
counter_different = 0

i = 0
while i < len(y_train_old):
    print("Step", i, "von", len(y_train_old))
    if y_train_old[i] == y_train_new[i]:
        flip.append(0)                      # store 0 if label is not flipped
        counter_same += 1
    else:
        flip.append(1)                      # store 1 if label is flipped
        counter_different += 1
    i += 1

print("same:", counter_same)
print("different:", counter_different)
print("total:", (counter_same+counter_different))
        
pickle.dump(flip, open('../applications/noisy_label/flip.pkl', 'wb'))

