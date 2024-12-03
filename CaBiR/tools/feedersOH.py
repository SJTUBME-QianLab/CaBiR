from __future__ import print_function
import numpy as np
import pandas as pd
import os
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset

# from tools.catch import CatchImpute


def MyInt(s):
    close = int(np.round(s))
    if abs(s - close) < 1e-6:
        return close
    else:
        return None


def OneHot(data0):
    onehot_cols = [
        'MRIMANUj', 'MRIMODLj', 'NACCADC', 'MRIMANU', 'MRIMODL',
        'MANU', 'MODL', 'SITE_ID',
        'sex', 'hemisphere', 'gender', 'batch',
    ]
    train_val = pd.concat([data0['train'], data0['val']], axis=0)
    train_val1 = pd.DataFrame(index=train_val.index)
    for col in train_val.columns:
        all_values = [np.nan if np.isnan(x) else MyInt(x) for x in train_val[col].unique()]
        print(all_values)
        if None in all_values or col == 'MRIFIELD':
            print(f'{col} is float')
            assert col not in onehot_cols
            train_val1 = pd.concat([train_val1, train_val[col]], axis=1)
            value = np.nanmean(train_val[col].values)
            train_val1[col].fillna(value, inplace=True)
        else:
            print(f'{col} is int')
            assert col in onehot_cols
            if np.nan in all_values:
                all_values.remove(np.nan)
            dummy = pd.get_dummies(train_val[col], prefix=col, dummy_na=False)
            # print(dummy.iloc[0, :])
            train_val1 = pd.concat([train_val1, dummy], axis=1)
            print(list(dummy.columns), [f'{col}_{x}' for x in sorted(all_values)])
            assert [ll.split('.')[0] for ll in dummy.columns] == [f'{col}_{x}' for x in sorted(all_values)]
    data1 = dict()
    for kk in data0.keys():
        if kk in ['train', 'val']:
            data1[kk] = train_val1.loc[data0[kk].index].values
        else:
            data1[kk] = pd.DataFrame(index=data0[kk].index).values
    return data1, train_val1.columns


def preprocess(data0, impute, normalize):
    train = data0['train']
    data = train.values  # sample * feature

    if impute == 'mean':
        value = np.nanmean(data, axis=0, keepdims=True)
    elif impute == 'median':
        value = np.nanmedian(data, axis=0, keepdims=True)
    else:
        raise NotImplementedError(f'Imputation method {impute} is not implemented')
    data1 = dict()
    for kk in data0.keys():
        data1[kk] = np.where(np.isnan(data0[kk].values), value, data0[kk].values)

    if normalize.lower() == 'std':
        STD = np.nanstd(data, axis=0, keepdims=True)
        MEAN = np.nanmean(data, axis=0, keepdims=True)
        for kk in data1.keys():
            data1[kk] = np.where(STD != 0, (data1[kk] - MEAN) / STD, 0)
    elif normalize.lower() == 'minmax':
        MAX = np.nanmax(data, axis=0, keepdims=True)
        MIN = np.nanmin(data, axis=0, keepdims=True)
        for kk in data1.keys():
            data1[kk] = np.where(MAX != MIN, (data1[kk] - MIN) / (MAX - MIN), 0.5)
    elif normalize is None:
        pass
    else:
        raise NotImplementedError('Normalization method should be in [Std, MinMax]')

    return data1


class Feeder(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data.astype(np.float32)
        self.dims = self.data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return data


class FeederPKL(Dataset):
    def __init__(self, data_config, raw, fold=0, subset='train'):
        super().__init__()
        assert subset in ['train', 'val', 'test']
        self.pkl_path = data_config.pop('pkl_path')
        assert len(data_config) == 0
        self.fold = fold
        self.subset = subset

        self.label, self.subject = self.cut_data(raw)
        with open(self.pkl_path, 'rb') as f:
            images = pickle.load(f)
        self.image = [images[sub].astype(np.float32) for sub in self.subject]
        self.dims = list(self.image[0].shape)

    def cut_data(self, raw):
        label_df, foldsID = raw
        if foldsID is None:
            subject = label_df.columns
        else:
            subject = np.intersect1d(foldsID[f'{self.subset}{self.fold}'], label_df.columns)
        label = label_df[subject].values[0]
        return label, subject

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        label = self.label[index]
        image = self.image[index]
        return np.expand_dims(image, axis=0), label, index


class FeederCSV(Dataset):
    def __init__(self, data_config, raw, fold=0, subset='train'):
        assert len(data_config) == 0
        assert subset in ['train', 'val', 'test']
        self.fold = fold
        self.subset = subset

        self.label, self.subject = self.cut_data(raw)
        self.data = None

    def cut_data(self, raw):
        label_df, foldsID = raw
        if foldsID is None:
            subject = label_df.columns
        else:
            subject = np.intersect1d(foldsID[f'{self.subset}{self.fold}'], label_df.columns)
        label = label_df[subject].values[0]
        return label, subject

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label, index


class FeederAdj(Dataset):
    def __init__(self, data_config, raw, fold=0, subset='train'):
        super().__init__()
        assert subset in ['train', 'val', 'test']
        self.pkl_path = data_config.pop('pkl_path')
        assert len(data_config) == 0
        self.fold = fold
        self.subset = subset

        self.label, self.subject = self.cut_data(raw)
        with open(self.pkl_path, 'rb') as f:
            graph = pickle.load(f)
        self.node = [graph[sub][0].astype(np.float32) for sub in self.subject]
        self.adj = [graph[sub][1].astype(np.float32) for sub in self.subject]
        self.dims = self.node[0].shape[1]

    def cut_data(self, raw):
        label_df, foldsID = raw
        if foldsID is None:
            subject = label_df.columns
        else:
            subject = np.intersect1d(foldsID[f'{self.subset}{self.fold}'], label_df.columns)
        label = label_df[subject].values[0]
        return label, subject

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        label = self.label[index]
        node = self.node[index]
        adj = self.adj[index]
        stack = np.concatenate([adj, node], axis=1)
        return stack, label, index


def get_sub_index(subset, fold, foldsID, all_subject):
    assert subset in ['train', 'val', 'test']
    sub_index = [all_subject.index(ss) for ss in foldsID[f'{subset}{fold}']]
    return sub_index

