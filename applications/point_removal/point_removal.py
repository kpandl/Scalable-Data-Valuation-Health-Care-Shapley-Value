from applications.point_removal.plot_densenet import *
import numpy as np


# define order
#HtoL = True         # removing data from highest to lowest

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
sx_train = X_train[:100000]
sy_train = y_train[:100000]
sx_val = X_test[10000:20000]
sy_val = y_test[10000:20000]


print("train_size:", sx_train.shape)
print("test_size:", sx_val.shape)


# load knn shapley values
with open('../applications/point_removal/deep_features_knn_new.npz', 'rb') as f:
    df_knn_sv = np.load(f)["knn"]
print("knn sv shape: ", df_knn_sv.shape)


k = 6
x_ratio = 0.1
count = int(len(sx_train)/2)
interval = int(count * x_ratio)
x_arrange = np.arange(0, count, interval)/len(sx_train) * 100
#count = int(len(sx_train))
#interval = int(count * x_ratio)
#x_arrange = np.arange(0, count, interval)

# perform experiment
random_auc = eval_densenet_sum_random(sx_train, sy_train, sx_val, sy_val, x_ratio, count, batch_size = 8, epochs=3)
df_HtoL_auc = eval_densenet_sum_single(df_knn_sv, sx_train, sy_train, sx_val, sy_val, x_ratio, count, batch_size=8, epochs=3, HtoL=True)
df_LtoH_auc = eval_densenet_sum_single(df_knn_sv, sx_train, sy_train, sx_val, sy_val, x_ratio, count, batch_size=8, epochs=3, HtoL=False)

print("-----------------------------------------------------------------------")
print("Final results:", "\n", "--Random:", random_auc, "\n", "--High to Low:", df_HtoL_auc, "\n", "--Low to High:", df_LtoH_auc)
print("-----------------------------------------------------------------------")

# save values
np.savez('../applications/point_removal/val_result_HtoL.npz', x=x_arrange, random=random_auc, HtoL=df_HtoL_auc, LtoH=df_LtoH_auc)


