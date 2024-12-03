import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, DenseGraphConv, DenseGINConv, DenseSAGEConv


class LayerRD(nn.Module):
    def __init__(self, n_inputs, n_outputs, GLayer, dropout=0.2, relu='leaky'):
        super().__init__()
        if GLayer == "DenseGINConv":
            self.layer = DenseGINConv(nn.Linear(n_inputs, n_outputs))
        else:
            self.layer = eval(GLayer)(n_inputs, n_outputs)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU() if relu == 'leaky' else nn.ReLU()

    def forward(self, x, adj):  # Linear -> ReLu -> Dropout
        x = self.layer(x, adj)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class GNNsRDCat2(nn.Module):
    def __init__(self, n_inputs, hidden, GLayer, dropout=0.2, relu='leaky'):
        super().__init__()
        self.depth = len(hidden)  # [16, 32], depth=3
        assert self.depth >= 1
        hidden = [n_inputs] + hidden
        self.dim = sum(hidden) * 2

        self.networks = nn.ModuleDict()
        for i in range(self.depth):  # layer1, layer2
            self.networks.add_module(
                f'layer{i}',
                LayerRD(hidden[i], hidden[i + 1], GLayer, dropout, relu))

    def forward(self, x):
        bsz, N, _ = x.shape
        adj = x[:, :, :N]
        x = x[:, :, N:]
        xs = [x]
        for i in range(self.depth):
            x = self.networks[f'layer{i}'](x, adj)
            xs.append(x)
        return torch.cat([readout(xx) for xx in xs], dim=-1)


def readout(x):
    return torch.concat([global_max_pool(x), global_mean_pool(x)], dim=-1)


def global_max_pool(x):  # x:[B,P,d]
    return F.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze(2)


def global_mean_pool(x):  # x:[B,P,d]
    return F.avg_pool1d(x.transpose(1, 2), x.shape[1]).squeeze(2)
