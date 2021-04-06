import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

x = np.array([10, 100, 200, 400, 800, 1000, 5000, 10000, 20000, 50000, 100000])


'''
knn = np.array([0.0769142468770345,   0.677141539255778,   1.653036856651306,   3.4390464584032694,  8.59050339460373,    12.708731484413146]) * 60
loo = np.array([0.7347790956497192,   66.44814310471217]) * 60
tmc = np.array([11.529986302057901]) * 60
g = np.array([0.12539432843526205, 0.9315359711647033, 3.903498136997223, 9.672818299134573, 50.83118432760239,150.22751605113348]) * 60
'''

# alternatively import computing results here
knn = np.array([4, 7, 11, 20, 45, 63, 900, 3362, 16525, 88450])
loo = np.array([28, 507, 1617, 5753, 22083, 33676])
tmc = np.array([5284])



plt.loglog(x[0:loo.shape[0]], loo, '^-', color = 'olive', label = "Leave-One-Out")
plt.loglog(x[0:tmc.shape[0]], tmc, 's-', color = 'navy', label = "TMC-Shapley")
plt.loglog(x[0:knn.shape[0]], knn, 'o-', color='darkorange', label = 'KNN-Shapley')
#plt.loglog(x[0:g.shape[0]], g, 's-', color = 'purple', label = "G-Shapley")


plt.xlabel('Number of training data points in log scale' + '\n')
plt.ylabel('Running time in log scale (s)')
plt.subplots_adjust(bottom=0.15)
plt.legend(loc='lower right')
plt.show()