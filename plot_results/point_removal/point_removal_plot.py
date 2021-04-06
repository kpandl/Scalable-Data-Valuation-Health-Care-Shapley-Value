import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

x = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
# alternatively import computing results here
'''
with open('val_result_HtoL.npz', "rb") as f:
    random_auc = np.load(f)["random"]
    HtoL_auc = np.load(f)["HtoL"]
    LtoH_auc = np.load(f)["LtoH"]

'''
# alternatively import computing results here
random_auc = np.array([0.8557, 0.8535, 0.8524, 0.8540, 0.8538, 0.8463, 0.8534, 0.8448, 0.8489, 0.8509])
HtoL_auc = np.array([0.8546, 0.8528, 0.8480, 0.8349, 0.8367, 0.8211, 0.8154, 0.7785, 0.7289, 0.6919])
LtoH_auc = np.array([0.8512, 0.8528, 0.8530, 0.8531, 0.8515, 0.8470, 0.8518, 0.8462, 0.8522, 0.8428])




#plt.plot(x, random_auc, '^-', color = 'olive', label = "Remove data randomly")
plt.plot(x, random_auc, color = 'olive', linestyle="dashed", label = "Remove data randomly")
plt.plot(x, HtoL_auc, 's-', color = 'navy', label = "Remove data of high  value")
plt.plot(x, LtoH_auc, 'o-', color='darkorange', label = 'Remove data of low value')


plt.xlabel('Fraction of train data removed (%)')
plt.ylabel('Prediction score')
plt.subplots_adjust(bottom=0.15)
plt.legend(loc='lower right')
plt.show()