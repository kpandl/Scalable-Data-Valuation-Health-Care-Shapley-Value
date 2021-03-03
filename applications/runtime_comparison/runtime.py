import time
import numpy as np
from applications.runtime_comparison.DShap import DShap
import math
import pickle



# load raw train and test data
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


# load train and test deep features
for i in range(1):
    with open('../arrays/deep_features/train/df_' + str(i) + '.npz', "rb") as f:
        x = np.load(f)["x"]
        y = np.load(f)["y"]
        if i == 0:
            X_train_deep = x
            y_train_deep = y
        else:
            X_train_deep = np.concatenate((X_train_deep, x), axis=0)
            y_train_deep = np.concatenate((y_train_deep, y), axis=0)
print("Train deep features:", X_train_deep.shape, y_train_deep.shape)

for i in range(1):
    with open('../arrays/deep_features/test/df_' + str(i) + '.npz', "rb") as f:
        x = np.load(f)["x"]
        y = np.load(f)["y"]
        if i == 0:
            X_test_deep = x
            y_test_deep = y
        else:
            X_test_deep = np.concatenate((X_test_deep, x), axis=0)
            y_test_deep = np.concatenate((y_test_deep, y), axis=0)
print("Test deep features:", X_test_deep.shape, y_test_deep.shape)


# define train size and time lists
train_size = [10, 100, 200, 400, 800, 1000, 5000, 10000, 20000, 50000, 100000]
time_knn = []
time_tmc = []
time_loo = []
time_g = []

# parameters
directory = '/pfs/data5/home/kit/aifb/dh9881/data_valuation/temp_runtime/'
store_data = '/pfs/data5/home/kit/aifb/dh9881/data_valuation/temp_runtime/data'
model_family = 'DenseNet'
metric = 'auc'

for size in train_size:
    # calculate values for different train size
    train_num = size
    test_num = math.ceil((size/0.8) - size)
    print('train_num:', train_num)
    print('test_num:', test_num)
    x_tr = X_train[0:train_num].astype("float32")
    y_tr = y_train[0:train_num].astype("float32")
    x_te = X_test[0:test_num].astype("float32")
    y_te = y_test[0:test_num].astype("float32")
    x_tr_deep = X_train_deep[0:train_num].astype("float32")
    x_te_deep = X_test_deep[0:train_num].astype("float32")
    
    print('---1. calculate knn run time')
    start_time = time.time()
    dshap = DShap(x_tr, y_tr, x_te, y_te, model_family, metric, X_train_deep=x_tr_deep, X_test_deep=x_te_deep, directory=directory)
    dshap.run(10, 0.5, knn_run=True)
    time_knn.append(str((time.time() - start_time)))
    print("--- %s seconds ---" % ((time.time() - start_time)))
    print('knn time:', time_knn)
    f = open(store_data+'knn_time.pkl', 'wb')
    data = {'knn_runtime': time_knn, 'train_size': train_size}
    pickle.dump(data, f)
    f.close()

    print('---2. calculate loo run time')
    start_time = time.time()
    dshap = DShap(x_tr, y_tr, x_te, y_te, model_family, metric, directory=directory)
    dshap.run(10, 0.5, loo_run=True)
    time_loo.append(str((time.time() - start_time)))
    print("--- %s seconds ---" % ((time.time() - start_time)))
    print('time loo:', time_loo)
    f = open(store_data+'loo_time.pkl', 'wb')
    data = {'loo_runtime': time_loo, 'train_size': train_size}
    pickle.dump(data, f)
    f.close()
    
    print('---3. calculate tmc run time')
    start_time = time.time()
    dshap = DShap(x_tr, y_tr, x_te, y_te, model_family, metric, directory=directory)
    dshap.run(10, 0.5, tmc_run=True)
    time_tmc.append(str((time.time() - start_time)))
    print("--- %s seconds ---" % ((time.time() - start_time)))
    print('time tmc:', time_tmc)
    f = open(store_data+'tmc_time.pkl', 'wb')
    data = {'tmc_runtime': time_tmc, 'train_size': train_size}
    pickle.dump(data, f)
    f.close()

