import torch
import torch.nn as nn
import torch.nn.functional as F
if_print = False


class ConvLayerCBRD(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel, BN=True, dropout=0.2, relu='leaky'):
        super(ConvLayerCBRD, self).__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        self.conv = nn.Conv3d(n_inputs, n_outputs,
                              kernel_size, kernel_stride, kernel_padding,
                              bias=False)
        if BN:
            self.BN = nn.BatchNorm3d(n_outputs)
        else:
            self.BN = None
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU() if relu == 'leaky' else nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.BN:
            x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class CNNRectangle(nn.Module):
    def __init__(self, n_inputs=1, width=16, depth=3, BN=True, dropout=0.2, relu='leaky', kernel=None):
        super(CNNRectangle, self).__init__()
        if kernel is None:
            kernel = [[7, 3, 0], [4, 2, 0]] + [[3, 1, 0]] * (depth - 2)
        assert depth == len(kernel) and isinstance(kernel[0], list)
        self.depth = depth

        self.networks = nn.ModuleDict()
        for i in range(depth):
            if i == 0:
                in_dim = n_inputs
            else:
                in_dim = width
            self.networks.add_module(
                f'layer{i}',
                ConvLayerCBRD(in_dim, width, kernel[i], BN, dropout, relu)
            )

    def forward(self, x):
        for i in range(self.depth):
            x = self.networks[f'layer{i}'](x)
            if if_print:
                print(x.shape)
        x = F.adaptive_avg_pool3d(x, output_size=(1, 1, 1))
        x = torch.flatten(x, 1)
        return x

    def size(self, in_dims=(182, 218, 182)):
        x = torch.ones([1, 1] + list(in_dims))
        x = self.forward(x)
        return x.shape[1]


class CNNTriangle(nn.Module):
    def __init__(self, n_inputs=1, width=16, depth=3, BN=True, dropout=0.2, relu='leaky', kernel=None):
        super(CNNTriangle, self).__init__()
        if kernel is None:
            kernel = [[7, 3, 0], [4, 2, 0]] + [[3, 1, 0]] * (depth - 2)
        assert depth == len(kernel) and isinstance(kernel[0], list)
        self.depth = depth

        self.networks = nn.ModuleDict()
        for i in range(depth):
            if i == 0:
                in_dim = n_inputs
                out_dim = width
            else:
                in_dim = width * 2 ** (i - 1)
                out_dim = width * 2 ** i
            self.networks.add_module(
                f'layer{i}',
                ConvLayerCBRD(in_dim, out_dim, kernel[i], BN, dropout, relu)
            )

    def forward(self, x):
        for i in range(self.depth):
            x = self.networks[f'layer{i}'](x)
            if if_print:
                print(x.shape)
        x = F.adaptive_avg_pool3d(x, output_size=(1, 1, 1))
        x = torch.flatten(x, 1)
        return x

    def size(self, in_dims=(182, 218, 182)):
        x = torch.ones([1, 1] + list(in_dims))
        x = self.forward(x)
        return x.shape[1]

