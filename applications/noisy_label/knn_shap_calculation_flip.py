import numpy as np
from applications.noisy_label.utils import *
import time


# load train and test deep features
for i in range(10):
    with open('../applications/noisy_label/flip_arrays/deep_features/train/df_' + str(i) + '.npz', "rb") as f:
        x = np.load(f)["x"]
        y = np.load(f)["y"]
        if i == 0:
            X_train_deep = x
            y_train_deep = y
        else:
            X_train_deep = np.concatenate((X_train_deep, x), axis=0)
            y_train_deep = np.concatenate((y_train_deep, y), axis=0)

for i in range(10):
    with open('../applications/noisy_label/flip_arrays/deep_features/test/df_' + str(i) + '.npz', "rb") as f:
        x = np.load(f)["x"]
        y = np.load(f)["y"]
        if i == 0:
            X_test_deep = x
            y_test_deep = y
        else:
            X_test_deep = np.concatenate((X_test_deep, x), axis=0)
            y_test_deep = np.concatenate((y_test_deep, y), axis=0)

# limit number of data points due to computational resource limits
X_train_deep = X_train_deep[:100000]
y_train_deep = y_train_deep[:100000]
X_test_deep = X_test_deep[:5000]
y_test_deep = y_test_deep[:5000]


# calculate values
k = 6

start_time = time.time()
print("------------Start:", start_time)
deep_knn_values, *_ = old_knn_shapley(k, X_train_deep, X_test_deep, y_train_deep, y_test_deep)
np.savez_compressed('../applications/noisy_label/deep_features_knn.npz', knn=deep_knn_values)
print("------------Dauer:", (time.time() - start_time))

