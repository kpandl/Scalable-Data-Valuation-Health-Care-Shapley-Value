import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# load array that stores information about flipped data
flip = pickle.load(open("../applications/noisy_label/flip_new.pkl", "rb"))
flip = np.array(flip)


# load KNN values
with open('../applications/noisy_label/deep_features_knn.npz', 'rb') as file:
    knn_v = np.load(file)["knn"]

# sort data points according to the KNN values
knn_i = np.argsort(-knn_v)[::-1]

cnt = 0
f = []
total = 0
cnt = 0

# go through flip array according to KNN values and identify flipped labels
for i in range(len(knn_i)):
    if flip[int(knn_i[i])] == 1:
        total += 1
for i in range(len(knn_i)):
    if flip[int(knn_i[i])] == 1:
        cnt += 1
    f.append(1.0 * cnt / total)
x = np.array(range(1, len(knn_i) + 1)) / len(knn_i) * 100
x = np.append(x[0:-1:10], x[-1])
f = np.append(f[0:-1:10], f[-1])

# store data and plot results
pickle.dump(x, open('../applications/noisy_label/x_knn.pkl', 'wb'))
pickle.dump(np.array(f), open('../applications/noisy_label/f_knn.pkl', 'wb'))
plt.plot(x, np.array(f) * 100, 'o-', color='purple', label = 'Selected by KNN-Shapley')



# random
ran_v = np.random.rand(len(knn_v))

# sort data points randomly
ran_i = np.argsort(-ran_v)[::-1]

cnt = 0
f = []
total = 0
cnt = 0

# go through flip array randomly and identify flipped labels
for i in range(len(ran_i)):
    if flip[int(ran_i[i])] == 1:
        total += 1
for i in range(len(ran_i)):
    if flip[int(ran_i[i])] == 1:
        cnt += 1
    f.append(1.0 * cnt / total)
x = np.array(range(1, len(ran_i) + 1)) / len(ran_i) * 100
f = x / 100

# store data and plot results
pickle.dump(x, open('../applications/noisy_label/x_random.pkl', 'wb'))
pickle.dump(np.array(f), open('../applications/noisy_label/f_random.pkl', 'wb'))
plt.plot(x, np.array(f) * 100, '--', color='red', label = "Selected randomly")


# general information
plt.xlabel('Fraction of data inspected (%)')
plt.ylabel('Fraction of incorrect labels (%)')
plt.legend(loc='lower right')
plt.show()



