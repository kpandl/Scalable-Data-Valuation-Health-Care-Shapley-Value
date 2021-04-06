from import_utils import *
import numpy as np


# open config
with open('config.json') as config_file:
    config = json.load(config_file)

# load training data
X_train, y_train = image_train_gen(config["path_to_train_csv"], length=150000, type=None)

# load test data
X_test, y_test = image_train_gen(config["path_to_test_csv"], length=37500, type=None)
print("Data loaded")

# save data
for i in range(10):
    np.savez_compressed("../arrays/raw_data/train/" + "df_" + str(i) + ".npz", x=X_train[i * 15000:i * 15000 + 15000], y=y_train[i * 15000:i * 15000 + 15000])
for i in range(10):
    np.savez_compressed("../arrays/raw_data/test/" + "df_" + str(i) + ".npz", x=X_test[i * 3750:i * 3750 + 3750], y=y_test[i * 3750:i * 3750 + 3750])
print("Data saved")