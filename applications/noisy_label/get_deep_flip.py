import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.nn import Module, BCELoss
from torch.optim import Adam
from applications.noisy_label.utils import *

# load model
model = DenseNet121()
# Compile optimizer
optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
# Loss
loss_criteria = BCELoss()
# Config device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Config progress bar
mb = master_bar(range(3))
mb.names = ['Training loss', 'Training AUROC']

# load train data
for i in range(10):
    with open('../applications/noisy_label/flip_arrays/raw_data/train/df_' + str(i) + '.npz', "rb") as f:
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
    with open('../applications/noisy_label/flip_arrays/raw_data/test/df_' + str(i) + '.npz', "rb") as f:
        x = np.load(f)["x"]
        y = np.load(f)["y"]
        if i == 0:
            X_test = x
            y_test = y
        else:
            X_test = np.concatenate((X_test, x), axis=0)
            y_test = np.concatenate((y_test, y), axis=0)
print("Test data:", X_test.shape, y_test.shape)

# transform ndarray to tensor
if isinstance(X_train, np.ndarray):
    X_train = torch.from_numpy(X_train).float()
if isinstance(y_train, np.ndarray):
    y_train = torch.from_numpy(y_train).float()
if isinstance(X_test, np.ndarray):
    X_test = torch.from_numpy(X_test).float()
if isinstance(y_test, np.ndarray):
    y_test = torch.from_numpy(y_test).float()
 
# create datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# create dataloaders
train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

for epoch in mb:
    # TRAINING
    model.train()
    train_epoch_loss = 0
    
    # run batches
    for train_batch, (train_images, train_labels) in enumerate(progress_bar(train_dataloader, parent=mb)):
        # move images, labels to device (GPU)
        train_images = train_images.to(device)  # 16, 3, 224, 224
        train_labels = train_labels.to(device)  # 16, 1
        # clear previous gradient
        optimizer.zero_grad()
        # feed forward the model
        train_pred = model(train_images, phase='train')
        train_batch_loss = loss_criteria(train_pred, train_labels)
        # back propagation
        train_batch_loss.backward()
        # update parameters
        optimizer.step()
        # update training loss after each batch
        train_epoch_loss += train_batch_loss.item()
        mb.child.comment = f'Training loss {train_epoch_loss / (train_batch + 1)}'
    # clear memory
    del train_images, train_labels, train_batch_loss
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    # calculate training loss
    train_final_loss = (train_epoch_loss / len(train_dataloader))
    mb.write('Finish training epoch {} with loss {:.4f}'.format(epoch, train_final_loss))
    
print("-------------Start obtaining deep features")
# train data
deep_features_X = []
deep_features_Y = []
for (images, labels) in train_dataloader:
    images, labels = images.to(device), labels.to(device)
    deep_feature = model(images, phase='deep')
    deep_features_X.append(deep_feature.cpu().detach().numpy())
    deep_features_Y.append(labels.cpu().detach().numpy())
    print(len(deep_features_X))
    
deep_features_image = np.concatenate(deep_features_X)
deep_features_label = np.concatenate(deep_features_Y)
print("deep features shape: ", deep_features_image.shape)
print("deep features shape: ", deep_features_label.shape)

for i in range(10):
    np.savez_compressed("../applications/noisy_label/flip_arrays/deep_features/train/" + "df_" + str(i) + ".npz", x=deep_features_image[i * 15000:i * 15000 + 15000],
                        y=deep_features_label[i * 15000:i * 15000 + 15000])
print("Deep features for train data saved")

# test data
deep_features_X = []
deep_features_Y = []
for (images, labels) in test_dataloader:
    images, labels = images.to(device), labels.to(device)
    deep_feature = model(images, phase='deep')
    deep_features_X.append(deep_feature.cpu().detach().numpy())
    deep_features_Y.append(labels.cpu().detach().numpy())
    print(len(deep_features_X))
    
deep_features_image = np.concatenate(deep_features_X)
deep_features_label = np.concatenate(deep_features_Y)

for i in range(10):
    np.savez_compressed("../applications/noisy_label/flip_arrays/deep_features/test/" + "df_" + str(i) + ".npz", x=deep_features_image[i * 3750:i * 3750 + 3750],
                        y=deep_features_label[i * 3750:i * 3750 + 3750])
print("Deep features for test data saved")