import random
from applications.point_removal.utils import *
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from fastprogress.fastprogress import master_bar, progress_bar



# GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def plot_train(phase, model, device, x_data, y_data, optimizer, criterion, batch_size, epochs=1):
    """fits the model with train data or evaluates model on test data

    # Arguments
            phase: training or validation
            model: ML model used for learning task
            device: device used, GPU or CPU
            X_data: data array of images (train or test)
            y_data: data array of label (train or test)
            optimizer: training optimizer
            criterion: loss criterion
            batch_size: number of samples per gradient update.
            epochs: number of times to iterate over the data arrays

    # Returns
            trained model or performance score"""
    
    # create tensors from numpy.ndarray
    if isinstance(x_data, np.ndarray):
        x_data = torch.from_numpy(x_data).float()
    if isinstance(y_data, np.ndarray):
        y_data = torch.from_numpy(y_data).float()
    # create datasets
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    # Config progress bar
    mb = master_bar(range(epochs))
    mb.names = ['Loss', 'AUROC score']

    for epoch in mb:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        pred_vals = torch.FloatTensor().to(device)  # tensor stores prediction values
        gt_vals = torch.FloatTensor().to(device)    # tensor stores groundtruth values


        # Iterate over data.
        for batch, (inputs, labels) in enumerate(progress_bar(dataloader, parent=mb)):
            # move labels and inputs to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # update groundtruth values
            gt_vals = torch.cat((gt_vals, labels), 0)
            # clear previous gradient
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                # feed forward the model
                output = model(inputs)
                # determine loss
                loss = criterion(output, labels)
                # update prediction values
                pred_vals = torch.cat((pred_vals, output), 0)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item()
            mb.child.comment = f'Training loss {running_loss / (batch + 1)}'
        # calculate epoch loss and epoch auc
        epoch_loss = (running_loss / len(dataloader))
        epoch_auc = roc_auc_score(gt_vals.to("cpu").detach().numpy(), pred_vals.to("cpu").detach().numpy())

        if phase == 'train':
            avg_loss = epoch_loss
            t_auc = epoch_auc
        elif phase == 'val':
            val_loss = epoch_loss
            val_auc = epoch_auc

    if phase == 'train':
        print('---Finish training with Loss: {:.4f} and AUC: {:.4f}'.format(avg_loss, t_auc))
        return
    elif phase == 'val':
        print('---Finish validation with Loss: {:.4f} and AUC: {:.4f}'.format(val_loss, val_auc))
        return val_auc, val_loss


def eval_densenet_sum_random(sx_train, sy_train, sx_test, sy_test, x_ratio, count, batch_size=8, epochs=3):
    """removes train data points randomly, trains and evaluates model

    # Arguments
            sx_train: train data images
            sy_train: train data labels
            sx_train: test data images
            sy_train: test data labels
            x_ratio, count: define interval
            batch_size: number of samples per gradient update.
            epochs: number of times to iterate over the data arrays

    # Returns
            validation scores"""
    
    interval = int(count * x_ratio)
    random_auc = []
    keep_idxs = np.arange(0, len(sx_train))
    random.shuffle(keep_idxs)

    for j in range(0, count, interval):
        print("Start calculations for", len(keep_idxs), "data points")
        if len(keep_idxs) == len(sx_train):
            x_train_keep, y_train_keep = sx_train, sy_train
        else:
            x_train_keep, y_train_keep = sx_train[keep_idxs], sy_train[keep_idxs]

        random_densenet = DenseNet121()
        random_densenet = random_densenet.to(device)
        optimizer = Adam(random_densenet.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
        criterion = nn.BCELoss()

        plot_train("train", random_densenet, device, x_train_keep, y_train_keep, optimizer, criterion, batch_size, epochs)
        auc, loss = plot_train("val", random_densenet, device, sx_test, sy_test, optimizer, criterion, batch_size)
        print("Random test AUC on", len(keep_idxs), "data points:", auc, "\n")
        random_auc.append(auc)
        keep_idxs = keep_idxs[:-interval]
    print("RANDOM RESULTS:", random_auc, "\n", "\n")
    return random_auc

def eval_densenet_sum_single(knn_value, sx_train, sy_train, sx_test, sy_test, x_ratio, count, batch_size=8, epochs=3, HtoL=False):
    """removes train data points according to KNN values, trains and evaluates model

        # Arguments
                sx_train: train data images
                sy_train: train data labels
                sx_train: test data images
                sy_train: test data labels
                x_ratio, count: define interval
                batch_size: number of samples per gradient update.
                epochs: number of times to iterate over the data arrays

        # Returns
                validation scores"""

    print("knn select data")
    interval = int(count * x_ratio)
    knn_auc = []
    idxs = np.argsort(knn_value)
    keep_idxs = idxs.tolist()

    for j in range(0, count, interval):
        print("Start calculations for", len(keep_idxs), "data points")
        if len(keep_idxs) == len(sx_train):
            x_train_keep, y_train_keep = sx_train, sy_train
        else:
            x_train_keep, y_train_keep = sx_train[keep_idxs], sy_train[keep_idxs]

        knn_densenet = DenseNet121()
        knn_densenet = knn_densenet.to(device)
        optimizer = Adam(knn_densenet.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
        criterion = nn.BCELoss()

        plot_train("train", knn_densenet, device, x_train_keep, y_train_keep, optimizer, criterion, batch_size, epochs)
        auc, loss = plot_train("val", knn_densenet, device, sx_test, sy_test, optimizer, criterion, batch_size)
        print("KNN test AUC on", len(keep_idxs), "data points:", auc, "\n")
        knn_auc.append(auc)
        if(HtoL == True):
            keep_idxs = keep_idxs[:-interval] # removing data from highest to lowest
        else:
            keep_idxs = keep_idxs[interval:] # removing data from lowest to highest
    print("KNN RESULTS:", knn_auc)
    return knn_auc
