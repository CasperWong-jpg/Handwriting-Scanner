import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1024, 64)
        self.output = nn.Linear(64, 36)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, numChannels=3, numOutputs=10):
        super().__init__()
        self.conv1 = nn.Conv2d(numChannels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 14 * 14, 84)
        self.fc2 = nn.Linear(84, numOutputs)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvNetComplex(nn.Module):
    """
    A CNN with two convolutional layers. Produces better learning but at higher resource costs
    """
    def __init__(self, numChannels=3, numOutputs=10):
        super().__init__()
        self.conv1 = nn.Conv2d(numChannels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.fc1 = nn.Linear(12 * 5 * 5, 84)
        self.fc2 = nn.Linear(84, numOutputs)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x