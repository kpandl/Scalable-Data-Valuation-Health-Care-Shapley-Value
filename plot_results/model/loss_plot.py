import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# create array with number of steps
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# create loss arrays - alternatively import computing results here
training = np.array([0.4564, 0.4282, 0.4123, 0.3986, 0.3797, 0.3575, 0.3264, 0.2894, 0.2495, 0.2098])
validation = np.array([0.3423, 0.3421, 0.2966, 0.3083, 0.3214, 0.3483, 0.3439, 0.3984, 0.4349, 0.4975])

plt.plot(x, training, 's-', color='navy', label='Training error')               # 's-'
plt.plot(x, validation, 'o-', color = 'darkorange', label='Validation error')   # 'o-'
plt.xlabel('Number of training epochs')
plt.ylabel('Error')
plt.legend(loc='lower right')
plt.show()