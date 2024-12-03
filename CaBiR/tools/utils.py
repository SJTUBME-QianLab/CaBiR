import os
import json
import random
import numpy as np
import re
import time
import torch
import hashlib
import importlib
from tools.catch import DEFAULT_Model, DEFAULT_Optimizer, DEFAULT_Scheduler


metric_set = {
    'clf': [
        'acc', 'accuracy', 'ppv', 'npv',
        'pre', 'precision', 'sen', 'sensitivity', 'recall', 'tpr', 'spe', 'specificity', 'tnr', 'fpr',
        'f1', 'f1_score', 'f1score',
        'auc'
    ],
    'reg': [
        'mae', 'mse', 'rmse',
        'r2',
        'Pear', 'corr'
    ],
}


def setting_seed(seed, setting_torch=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if setting_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html


def remove_key(config, key):
    if key in config.keys():
        del config[key]


def fill_default_config(config, default_config, flag=''):
    # parameter priority: argparse (string from command line) > config.json > DEFAULT_*
    if config is None:
        return None
    config = config.copy()
    if flag.startswith('root'):
        config = str2list(str2bool(config))
        default_config = str2list(str2bool(default_config))
    # -----------------------------------------------
    config = {kk: vv for kk, vv in config.items() if not kk.endswith('_')}
    default_config = {kk: vv for kk, vv in default_config.items() if not kk.endswith('_')}

    if 'name' not in config.keys() and 'name' in default_config.keys():
        config['name'] = default_config['name']

    if 'name' in config.keys():
        name = config['name']
        if 'name' in default_config.keys():
            default_config.pop('name')
        if flag in ['enc', 'bias', 'env']:
            default_config = fill_default_config(default_config, DEFAULT_Model[name], flag='')
        elif flag == 'optimizer':
            default_config = fill_default_config(default_config, DEFAULT_Optimizer[name], flag='')
        elif flag == 'scheduler':
            default_config = fill_default_config(default_config, DEFAULT_Scheduler[name], flag='')
        elif flag.startswith('algorithm'):
            pass
        else:
            raise ValueError(f'Unknown parameters {flag}')
        default_config['name'] = name
    else:
        pass

    diff_set = set(config.keys()) - set(default_config.keys())
    # print(config, default_config, flag)
    if flag in ['data', 'algorithm', 'loss', 'ampl', 'info']:
        pass
    elif flag.startswith('model'):
        assert diff_set - {'lr_c', 'bias', 'env', 'dim'} == set(), f'WRONG ARG {diff_set}'
    else:
        assert diff_set == set(), f'WRONG ARG {diff_set}'

    for k in default_config.keys():
        if k not in config.keys():
            config[k] = default_config[k]

        if isinstance(default_config[k], dict):  # k='algorithm', 'model', 'optimizer', 'scheduler', etc
            # assert isinstance(config[k], dict)
            if not isinstance(config[k], dict):
                raise ValueError(f"WRONG ARG {k} {config[k]}")
            config[k] = fill_default_config(config[k], default_config[k], flag=k)

    if flag.startswith('root'):
        if len(config['data']) == 2:
            assert '_v' in config['data']['csv_dir']
        elif len(config['data']) > 2:
            assert 'image' in config['data']['csv_dir']
            if 'pkl_path' in config['data'].keys():
                config['data']['pkl_path'] = config['data']['pkl_path'].strip('\'')
                assert not ('suffix' in config['data'].keys() and 'npy_dir' in config['data'].keys())
            else:
                assert not ('pkl_path' in config['data'].keys())

    return config


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        diff = (te - ts) * 1000
        if 'log_time' in kw and kw['log_time'] is not None:
            name = kw.get('log_name', method.__name__)
            kw['log_time'][name] = diff
        else:
            print('%r: %.3f ms' % (method.__name__, diff))
        return result
    return timed


def local_time(form="%Y-%m-%d %H:%M:%S"):
    if form is None:
        return time.asctime()
    elif 'f' in form:
        ts = time.time()
        form, ms_form = form.split('%S')
        ms = ms_form % ((ts - int(ts)) * 1000)
        return time.strftime(form + '%S', time.localtime(ts)) + ms
    else:
        return time.strftime(form)


def str2bool(v):
    if isinstance(v, dict):
        return {k: str2bool(v) for k, v in v.items()}

    if not isinstance(v, str):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif v.lower() in ('none', 'null', 'nan'):
        return None
    else:
        return v


def str2list(v):
    if isinstance(v, dict):
        return {k: str2list(v) for k, v in v.items()}

    if not isinstance(v, str):
        return v
    try:
        if v[0] == '[' and v[-1] == ']':
            return [str2list(kk) for kk in eval(v)]
        elif v[0] == '(' and v[-1] == ')':
            return tuple(str2list(kk) for kk in eval(v))
        else:
            return v
    except SyntaxError:
        return v


def parse_label(task):
    if task.startswith('clf_'):
        pattern = r"([a-zA-Z]+)(\d)"
        groups = re.findall(pattern, task.split('clf_')[1])
        Label = {s: int(i) for (s, i) in groups}
        num_class = len(Label)
    elif task.startswith('reg_'):
        Label = None
        num_class = 1
    elif task == 'MotV1':
        Label = {'CT': 0, 'ET': 1, 'IT': 2, 'Lamp5': 3, 'NP': 4, 'Pvalb': 5, 'Sncg': 6, 'Sst': 7, 'Vip': 8}
        num_class = len(Label)
    elif task == 'MotV2':
        Label = {'CT': 0, 'ET': 1, 'IT': 2, 'Lamp5': 3, 'Pvalb': 4, 'Sst': 5, 'Vip': 6}
        num_class = len(Label)
    elif task == 'VisV1':
        Label = {'Lamp5': 0, 'Pvalb': 1, 'Serpinf1': 2, 'Sncg': 3, 'Sst': 4, 'Vip': 5}
        num_class = len(Label)
    else:
        raise ValueError(f"Unknown task: {task}")
    return num_class, Label


def parse_fuse(fuse_str, num_class):
    if fuse_str.startswith('train'):
        fuse = 'train'
        metric_name = fuse_str.split('_')[1]
        if num_class > 2:
            ave = fuse_str.split('_')[2]
        else:
            ave = None
    else:
        fuse = fuse_str
        metric_name = None
        ave = None
    return fuse, metric_name, ave


def parse_work_dir(config):
    save_config = config.copy()
    split_mode, split, fold, seed, _, _ = \
        [save_config.pop(kk) for kk in ['split_mode', 'split', 'fold', 'seed', 'save_dir', 'phase']]
    if 'test_data' in save_config.keys():
        save_config.pop('test_data')
    config_hash = hashlib.md5(json.dumps(save_config, sort_keys=True).encode('utf-8')).hexdigest()
    pattern = f"{split_mode[0]}.split{split}", f"{config_hash}_F{fold}S{seed}"
    return pattern, json.dumps(save_config, sort_keys=True)


def find_dir(folder, pattern, print_info=True):
    """
    :param folder: dir to find
    :param pattern: pattern to match
    :param print_info: whether to print information
    :return: the only one path that matches the pattern
    """
    if not os.path.isdir(folder):
        if print_info:
            print(f"Folder {folder} does not exist.")
        return []
    right = []
    for name in os.listdir(folder):
        m = re.search(pattern, name)
        if m is None:
            continue
        path = os.path.join(folder, name)
        right.append(path)
    return right


def find_work_dir(folder, pattern, finishline='Finish.\n', print_info=True):
    right = find_dir(folder, pattern, print_info)
    if len(right) == 1:
        path = right[0]
        try:
            file = open(os.path.join(path, 'log.txt'), 'r')
            # assert file.readlines()[-1].endswith(finishline)
            assert finishline in file.read()
            assert os.path.exists(os.path.join(path, 'trend_metrics_test.csv'))
            return path
        except FileNotFoundError:
            return 'NOT_' + path
        except AssertionError:
            return 'NOT_' + path
    elif len(right) > 1:
        if print_info:
            print(f"[>=1] \tfolder matches \"{pattern}\" in {folder}")
        return False
    else:
        if print_info:
            print(f"[0] \tfolder matches \"{pattern}\" in {folder}")
        return False


def remove_files(folder, pattern):
    files = find_dir(folder, pattern)
    for file in files:
        os.remove(file)


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])  # import return model
    mod = getattr(mod, components[1])
    return mod


def select_feeder(data_config, feeder):
    if 'pkl_path' in data_config and '_adj.pkl' in data_config['pkl_path']:
        Feeder0 = import_class(feeder + 'Adj')
    elif 'pkl_path' in data_config:
        Feeder0 = import_class(feeder + 'PKL')
    else:
        raise NotImplementedError(data_config)
    return Feeder0


def print_format(dict_values, keys=None, form='%.3fms'):
    if keys is None:
        keys = dict_values.keys()
    return ', '.join((f'[{k}] ' + form % dict_values[k]) for k in keys)


def random_select(label, k):
    select = []
    for c in sorted(set(label)):
        if label.tolist().count(c) < k:
            return None
        select.extend(random.sample([i for i, l in enumerate(label) if l == c], k))
    assert len(select) == len(set(select))
    assert [label[select].tolist().count(c) == k for c in set(label)]
    return select
