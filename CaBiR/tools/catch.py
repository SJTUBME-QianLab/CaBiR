import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
import torch
from backbone.Conv3dNet import *
from backbone.MultiLayer import *
from backbone.GraphNet import *


DEFAULT_Model = {
    "FullyConnectRBD": dict(
        BN=True, dropout=0.2,
    ),
    "MLPsRBD": dict(
        hidden=[16], BN=True, dropout=0.2,
    ),
    "CNNTriangle": dict(
        width=16, depth=3, BN=True, dropout=0.2,
        kernel=[[7, 3, 0], [4, 2, 0], [3, 1, 0]]
    ),
    'GNNsRDCat2': dict(
        hidden=[16], GLayer="DenseGraphConv", dropout=0.2, relu='leaky'
    )
}

DEFAULT_Optimizer = {
    "Adam": dict(
        lr=0.001,
        weight_decay=0
    ),
    "SGD": dict(
        lr=0.001,
        weight_decay=0,
        momentum=0,
        nesterov=False
    ),
}

DEFAULT_Scheduler = {
    "ReduceLROnPlateau": dict(
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        cooldown=0,
        min_lr=0,
    ),
    "StepLR": dict(
        step_size=10,
        gamma=0.1,
    ),
    "MultiStepLR": dict(
        milestones=[30, 80],
        gamma=0.1,
    ),
    "ExponentialLR": dict(
        gamma=0.1,
    ),
    "CosineAnnealingLR": dict(
        T_max=50,
        eta_min=0,
    ),
}


def CatchImpute(name):
    # all use default parameters
    # config = dict(keep_empty_features=True)
    if SimpleImputer.__init__.__code__.co_varnames.__contains__('keep_empty_features'):
        config = dict(keep_empty_features=True)
    else:
        config = dict()
    if name.lower() == 'mean':
        imp = SimpleImputer(strategy='mean', **config)
    elif name.lower() == 'median':
        imp = SimpleImputer(strategy='median', **config)
    elif name == 'most_frequent':
        imp = SimpleImputer(strategy='most_frequent', **config)
    elif name == 'constant':
        imp = SimpleImputer(strategy='constant', **config)
    elif name == 'KNN':
        imp = KNNImputer(**config)
    elif name == 'Multivariate':
        imp = IterativeImputer(random_state=2024, **config)
    else:
        raise ValueError(f'Imputation not supported: {name}')
    return imp


def CatchOptimizer(network, hparams):
    hparams = hparams.copy()
    name = hparams.pop('name')
    kwargs = {kk: hparams.pop(kk) for kk in DEFAULT_Optimizer[name].keys()}
    assert len(hparams) == 0, f'Invalid hparams: {hparams}'

    if name == 'Adam':
        optimizer = torch.optim.Adam(network.parameters(),  **kwargs)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(network.parameters(), **kwargs)
    else:
        raise ValueError(f'Optimizer not supported: {name}')
    return optimizer


def CatchScheduler(optimizer, hparams):
    hparams = hparams.copy()
    name = hparams.pop('name')
    kwargs = {kk: hparams.pop(kk) for kk in DEFAULT_Scheduler[name].keys()}
    assert len(hparams) == 0, f'Invalid hparams: {hparams}'

    if name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif name == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise ValueError(f'Scheduler not supported: {name}')
    assert len(hparams) == 0, f'Invalid hparams: {hparams}'
    return scheduler


def CatchNetwork(hparams):
    hparams = hparams.copy()
    name = hparams.pop('name')
    kwargs = {kk: hparams.pop(kk) for kk in DEFAULT_Model[name].keys()}

    if name in ['FullyConnectRBD', 'MLPsRBD']:
        network = eval(name)(
            n_inputs=hparams.pop('n_inputs'), n_outputs=hparams.pop('n_outputs'), **kwargs
        )
    elif name in ['CNNTriangle',]:
        network = eval(name)(n_inputs=1, **kwargs)
    elif name in ['GNNsRDCat2']:
        network = eval(name)(
            n_inputs=hparams.pop('n_inputs'), **kwargs
        )
    else:
        raise ValueError(f'Network not supported: {name}')
    assert len(hparams) == 0, f'Invalid hparams: {hparams}'

    return network

