import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerRBD(nn.Module):
    def __init__(self, n_inputs, n_outputs, BN=True, dropout=0.2, relu='leaky'):
        super(LayerRBD, self).__init__()
        self.layer = nn.Linear(n_inputs, n_outputs)
        if BN:
            self.BN = nn.BatchNorm1d(n_outputs)
        else:
            self.BN = None
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU() if relu == 'leaky' else nn.ReLU()

    def forward(self, x):  # Linear -> ReLu -> BatchNorm -> Dropout
        x1 = self.layer(x)
        x2 = self.relu(x1)
        if self.BN:
            x2 = self.BN(x2)
        x2 = self.dropout(x2)
        return x1, x2


class FullyConnectRBD(nn.Module):
    def __init__(self, n_inputs, n_outputs, BN=True, dropout=0.2, relu='leaky'):
        super(FullyConnectRBD, self).__init__()
        self.networks = LayerRBD(n_inputs, n_outputs, BN, dropout, relu)

    def forward(self, x, middle=False):
        x1, x = self.networks(x)
        if middle:
            return x1
        else:
            return x


class MLPsRBD(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden, BN=True, dropout=0.2, relu='leaky'):
        super(MLPsRBD, self).__init__()
        self.depth = len(hidden)  # [16, 32], depth=3

        self.networks = nn.ModuleDict()
        self.networks.add_module(
            'layer0',
            LayerRBD(n_inputs, hidden[0], BN, dropout, relu))
        for i in range(1, self.depth):  # layer1, layer2
            self.networks.add_module(
                f'layer{i}',
                LayerRBD(hidden[i - 1], hidden[i], BN, dropout, relu))
        self.output = nn.Linear(hidden[-1], n_outputs)

    def forward(self, x, middle=False):
        for i in range(self.depth):
            x1, x = self.networks[f'layer{i}'](x)
        if middle:
            return x1
        else:
            x = self.output(x)
            return x
