import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from fastprogress.fastprogress import master_bar, progress_bar

# model
class DenseNet(nn.Module):
    def __init__(self, num_classes=1):
        super(DenseNet, self).__init__()
        self.net = torchvision.models.densenet121(pretrained=True)
        # Feature extraction
        self.conv0 = self.net.features.conv0
        self.norm0 = self.net.features.norm0
        self.relu0 = self.net.features.relu0
        self.pool0 = self.net.features.pool0
        self.denseblock1 = self.net.features.denseblock1
        self.transition1 = self.net.features.transition1
        self.denseblock2 = self.net.features.denseblock2
        self.transition2 = self.net.features.transition2
        self.denseblock3 = self.net.features.denseblock3
        self.transition3 = self.net.features.transition3
        self.denseblock4 = self.net.features.denseblock4
        # Classification
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.classifier = nn.Sequential(nn.Linear(self.net.classifier.in_features, num_classes), nn.Sigmoid())

    def forward(self, inputs, phase):
        # Feature extraction
        conv = self.conv0(inputs)
        norm = self.norm0(conv)
        relu = self.relu0(norm)
        pool = self.pool0(relu)
        deep1 = self.denseblock1(pool)
        trans1 = self.transition1(deep1)
        deep2 = self.denseblock2(trans1)
        trans2 = self.transition2(deep2)
        deep3 = self.denseblock3(trans2)
        trans3 = self.transition3(deep3)
        deep4 = self.denseblock4(trans3)
        # Classification
        out = self.avgpool(deep4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        if phase == "deep":
            return deep4
        else:
            return out              #self.net(inputs)

    def score(self, X, y):
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        score = self.evaluate(X, y)
        return score
        


# GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def DenseNet121():
    """returns model"""
    return DenseNet().to(device)
    

def train_evaluate(phase, model, x_data, y_data, batch_size, epochs=1):
    """fits the model with train data or evaluates model on test data

        # Arguments
                phase: training or validation
                model: ML model used for learning task
                X_data: data array of images (train or test)
                y_data: data array of label (train or test)
                batch_size: number of samples per gradient update.
                epochs: number of times to iterate over the data flip_arrays

        # Returns
                trained model or performance score"""

    # define model, optimizer, criterion
    optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
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
        #mb.write('Finish epoch {} with loss {:.4f} and score {:.4f}'.format(epoch, epoch_loss, epoch_auc))

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


def old_knn_shapley(K, trainX, valX, trainy, valy):
    """calculates KNN-Shapley values

            # Arguments
                    K: number of nearest neighbors
                    trainX: deep features of train images
                    valX: deep features of test images
                    trainy: deep features of train labels
                    valy: deep features of test labels

            # Returns
                    KNN values"""

    N = trainX.shape[0]
    M = valX.shape[0]
    c = 1
    value = np.zeros(N)
    #     value = [[] for i in range(N) ]
    scores = []
    false_result_idxs = []
    for i in range(M):
        print("Step", i, "from", M)
        X = valX[i]
        y = valy[i]

        s = np.zeros(N)
        diff = (trainX - X).reshape(N, -1)  # calculate the distances between valX and every trainX data point
        dist = np.einsum('ij, ij->i', diff, diff)  # output the sum distance
        idx = np.argsort(dist)  # ascend the distance
        ans = trainy[idx]

        # calculate test performance
        score = 0.0

        for j in range(min(K, N)):
            score += float(ans[j] == y)
        if (score > min(K, N) / 2):
            scores.append(1)
        else:
            scores.append(0)
            false_result_idxs.append(i)

        s[idx[N - 1]] = float(ans[N - 1] == y) * c / N
        cur = N - 2
        for j in range(N - 1):
            s[idx[cur]] = s[idx[cur + 1]] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) * c / K * (
                        min(cur, K - 1) + 1) / (cur + 1)
            cur -= 1

        for j in range(N):
            value[j] += s[j]
    for i in range(N):
        value[i] /= M
    return value, np.mean(scores), false_result_idxs