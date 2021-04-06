import numpy as np
import torch
import torchvision
import torch.nn as nn


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

    def forward(self, inputs):
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
    