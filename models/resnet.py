import numpy as np
import torch
import torch.nn as nn
import torchvision
from applications.runtime_comparison.fit_module import FitModule



class ResNet(FitModule):
    def __init__(self, num_classes=1):
        super(ResNet, self).__init__()
        self.net = torchvision.models.resnet34(pretrained=True)
        kernel_count = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Linear(kernel_count, num_classes), nn.Sigmoid())

    def forward(self, inputs, phase):
        # Feature extraction
        conv1 = self.net.conv1(inputs)
        bn1 = self.net.bn1(conv1)
        relu = self.net.relu(bn1)
        maxpool = self.net.maxpool(relu)
        layer1 = self.net.layer1(maxpool)
        layer2 = self.net.layer2(layer1)
        layer3 = self.net.layer3(layer2)
        layer4 = self.net.layer4(layer3)
        # Classification
        out = self.net.avgpool(layer4)
        out = out.view(out.size(0), -1)
        out = self.net.fc(out)
        if phase == 'deep':
            return layer4
        else:
            return out

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


def ResNet34():
    """returns model"""
    return ResNet().to(device)








