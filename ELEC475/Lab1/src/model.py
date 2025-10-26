

import torch
import torch.nn.functional as F
import torch.nn as nn


class autoencoderMLP4Layer(nn.Module):

    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(autoencoderMLP4Layer, self).__init__()
        self.N2 = 392
        self.lin1 = nn.Linear(N_input, self.N2)
        self.lin2 = nn.Linear(self.N2, N_bottleneck)
        self.lin3 = nn.Linear(N_bottleneck, self.N2)
        self.lin4 = nn.Linear(self.N2, N_output)
        self.type = 'MLP4'
        self.input_shape = (1, 28*28)

    def forward(self, X):
        return self.decode(self.encode(X))
        X = self.lin1(X)
        X = F.relu(X)
        X = self.lin2(X)
        X = F.relu(X)
        X = self.lin3(X)
        X = F.relu(X)
        X = self.lin4(X)
        X = torch.sigmoid(X)
        return X

    def encode(self, X):
        X = self.lin1(X)
        X = F.relu(X)
        X = self.lin2(X)
        return F.relu(X)

    def decode(self, X):
        X = self.lin3(X)
        X = F.relu(X)
        X = self.lin4(X)
        X = torch.sigmoid(X)
        return X
