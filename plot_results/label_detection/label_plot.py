import pickle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set()

f_knn = pickle.load(open("f_knn.pkl", "rb"))
x_knn = pickle.load(open("x_knn.pkl", "rb"))
f_random = pickle.load(open("f_random.pkl", "rb"))
x_random = pickle.load(open("x_random.pkl", "rb"))


plt.plot(x_knn, f_knn, color='darkorange', linewidth=1.7, label='Selected by KNN-Shapley value')               # 's-'
plt.plot(x_random, f_random, color = 'olive', linestyle="dashed", linewidth=1.7, label='Selected randomly')   # 'o-'



plt.xlabel('Fraction of data inspected (%)')
plt.ylabel('Fraction of incorrect labels detected (%)')
plt.legend(loc='lower right')
plt.show()

