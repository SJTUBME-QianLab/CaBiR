from __future__ import print_function
import os
import shutil
import time
import pickle
from shutil import copytree, ignore_patterns
import numpy as np
import pandas as pd
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import shap

from tools import utils
from tools.evaluation import SettleResults

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class Processor:
    def __init__(self, config0, device=0):

        # Initialization
        utils.setting_seed(seed=config0['seed'], setting_torch=True)
        self.num_class, self.label2int = utils.parse_label(config0['task'])
        work_dir, self.config_str = utils.parse_work_dir(config0)
        phase = config0['phase']
        self.config = config0.copy()
        self.device = device
        data_config_test = self.config['test_data']
        data_config = self.config['data']
        data_config['split_package'] = work_dir[0]
        if all([data_config[kk] == data_config_test[kk] for kk in data_config_test.keys()]):
            data_config_test = None
        if phase == 'train':
            self.operator = 'train'
            # folders preparation to save all results
            # self.work_dir = os.path.join(self.config['save_dir'], work_dir[0] + f".{self.config['task']}",
            #                              work_dir[1])
            find = utils.find_work_dir(
                os.path.join(self.config['save_dir'], work_dir[0] + f".{self.config['task']}"),
                work_dir[1])
            # assert not find, f"Work dir {find} already exists"
            if find and find.startswith('NOT_'):
                shutil.rmtree(find.split('NOT_')[1])
            self.work_dir = os.path.join(self.config['save_dir'], work_dir[0] + f".{self.config['task']}",
                                         f"T{utils.local_time('%Y%m%d%H%M%S.%03.f')}_" + work_dir[1])
            os.makedirs(self.work_dir)
            # folders preparation to save model weights and results during training
            self.epoch_dir = os.path.join(self.work_dir, 'epoch')
            os.makedirs(self.epoch_dir)
            # folders preparation to save tensorboard and other logs
            self.tb_log_dir = os.path.join(self.work_dir, 'tb_log')
            os.makedirs(self.tb_log_dir)
            self.tb_writer = SummaryWriter(self.tb_log_dir)

            # backup code and config
            pwd = os.path.dirname(os.path.realpath(__file__))
            copytree(pwd, os.path.join(self.work_dir, 'code'), symlinks=False,
                     ignore=ignore_patterns('__pycache__', '*.csv'))
            with open(os.path.join(self.epoch_dir, 'config.json'), 'w') as file:
                json.dump(self.config, file, indent=4, sort_keys=True)
            self.print_log(f"String of hyper-parameters:\n{self.config_str}")

            self.global_step = 0
            self.best_metric = 0 if self.num_class > 1 else 1e5
            self.best_epoch = 0
        else:  # 模型已训练好，事后分析
            self.model_slc, self.operator = phase.split('.')
            assert self.model_slc in ['best', 'final', 'start']
            # self.work_dir = os.path.join(self.config['save_dir'], work_dir[0] + f".{self.config['task']}",
            #                              work_dir[1])
            # with open(os.path.join(self.work_dir, 'log.txt'), 'r') as file:
            #     if not file.readlines()[-1].endswith('Finish.\n'):
            #         raise FileNotFoundError('Training incomplete, cannot test')
            self.work_dir = utils.find_work_dir(
                os.path.join(self.config['save_dir'], work_dir[0] + f".{self.config['task']}"),
                work_dir[1])
            if not self.work_dir or self.work_dir.startswith('NOT_'):
                raise FileNotFoundError('Training incomplete, cannot test')
            self.work_dir = self.work_dir.split('NOT_')[1] if self.work_dir.startswith('NOT_') else self.work_dir
            print(self.work_dir)
            self.epoch_dir = os.path.join(self.work_dir, 'epoch')
            self.print_log('\n\n\n', print_time=False)

        self.print_log(phase + '-' * 10 + "Work dir: " + self.work_dir)
        self.log_time = dict()  # record time consumption
        self.print_log('Load data.')
        self.data_loader, mod2var, image_dims = self.load_data(data_config, data_config_test)

        self.print_log('Initialize algorithm.')
        algorithm = self.config['algorithm'].pop('name')
        algorithm = utils.import_class(algorithm)
        n_inputs = {mm: len(vv[0]) for mm, vv in mod2var.items()}
        self.print_log('n_inputs: ' + utils.print_format(n_inputs, form='%d') + ', [MRI] ' + str(image_dims))
        n_inputs['MRI'] = image_dims
        self.algorithm = algorithm(
            n_inputs=n_inputs, num_epochs=self.config['num_epochs'], num_class=self.num_class, hparams=self.config['algorithm']
        ).to(self.device)
        if self.operator == 'train':
            self.print_log(self.algorithm)
        self.print_log('groups_nonMRI: ' + utils.print_format({kk: n_inputs[kk] for kk in self.algorithm.groups_nonMRI}, form='%d'))

    def main(self):
        if self.operator == 'train':
            for epoch in range(self.config['num_epochs']):
                if epoch == 0:
                    self.algorithm.save_model(path=self.epoch_dir, epoch=0, flag='start')
                    self.print_log(f'Save epoch0_start_model.pth')

                self.train_an_epoch(epoch)

                if epoch % self.config['save_interval'] == 0:
                    with torch.no_grad():
                        self.valid_an_epoch(epoch, subset='val')
                # self.print_log(utils.print_format(self.log_time))

                if (self.best_epoch > int(self.config['num_epochs'] * 0.5) and
                        (epoch - self.best_epoch) > self.config['early_stop']):
                    self.print_log(f'Early stop at epoch {epoch}')
                    break

            self.algorithm.save_model(path=self.epoch_dir, epoch=epoch, flag='final')
            self.print_log(f'Save epoch{epoch}_final_model.pth')

            with torch.no_grad():
                self.valid_an_epoch(epoch=epoch, subset='val')
                # self.valid_an_epoch(epoch=epoch, subset='test')
                self.post_evaluate(phase=f'allepo_{epoch}.{epoch+1}_test')

            self.settle_results(epoch, ['val', 'test'])
            self.print_log('Max GPU mem = %.1f MB' % (torch.cuda.max_memory_allocated() / (1024. ** 2)))
            self.print_log(utils.print_format(self.log_time))

            if 'MRI_indi' in self.data_loader.keys():
                with torch.no_grad():
                    self.post_evaluate(phase='final.T_indi')

            self.print_log('Finish.')

        elif self.operator.startswith('T_'):
            self.algorithm.load_model(path=self.epoch_dir, select=self.model_slc)
            self.algorithm.set_train_status(False)
            with torch.no_grad():
                self.post_evaluate(phase=self.config['phase'])
            self.print_log('Finish.')

        elif 'CAM' in self.operator:
            self.algorithm.load_model(path=self.epoch_dir, select=self.model_slc)
            self.algorithm.set_train_status(False)
            self.post_cam(phase=self.config['phase'])
            self.print_log('Finish.')

        elif self.operator.startswith('shap'):
            self.algorithm.load_model(path=self.epoch_dir, select=self.model_slc)
            self.algorithm.set_train_status(False)
            self.post_shap(phase=self.config['phase'])
            self.print_log('Finish.')

        elif self.operator == 'emb':
            self.algorithm.load_model(path=self.epoch_dir, select=self.model_slc)
            self.algorithm.set_train_status(False)
            with torch.no_grad():
                self.post_emb(phase=self.config['phase'])
            self.print_log('Finish.')

        # together
        elif self.operator == 'together':
            if self.model_slc in ['final']:
                self.algorithm.load_model(path=self.epoch_dir, select=self.model_slc)
                self.algorithm.set_train_status(False)
                phase_head = self.config['phase'].split('together')[0]
                with torch.no_grad():
                    for subset in ['train', 'val', 'test'] + (['indi'] if 'MRI_indi' in self.data_loader.keys() else []):
                        self.post_evaluate(phase=phase_head + 'T_' + subset)  # final.T_indi
                    self.post_emb(phase=phase_head + 'emb')
                # self.post_cam(phase=phase_head + 'CAM')
                # self.post_shap(phase=phase_head + 'shap_5')
                self.print_log('Finish.')
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

    def timeit(method):
        def timed(self, *args, **kw):
            ts = time.time()
            result = method(self, *args, **kw)
            te = time.time()
            diff = (te - ts) * 1000
            name = method.__name__
            if name in self.log_time.keys():
                self.log_time[name] += diff
            else:
                self.log_time[name] = diff
            # self.print_log('%r: %.3f ms' % (name, diff))
            return result
        return timed

    @timeit
    def load_data(self, data_config, data_config_test=None):
        split_package = data_config.pop('split_package')

        csv_dir = data_config.pop('csv_dir')
        modality = pd.read_csv(os.path.join(csv_dir, 'Stats.csv'))
        Data = pd.read_csv(os.path.join(csv_dir, 'Data.csv'), index_col=0)
        if self.label2int:
            counts = Data['Label'].value_counts()
            self.print_log('Labels: ' + ', '.join([f'{kk}({self.label2int[kk]}):{counts[kk]}' for kk in counts.keys()]))
            Data['Label'] = Data['Label'].apply(lambda x: self.label2int.get(x))
            Data.index = Data.index.astype(str)
            Data = Data.dropna(subset=['Label'], axis=0, how='any')
        self.label_df = pd.DataFrame(dict(zip(Data.index, Data['Label'])), index=['true_label'])

        # 读取测试数据，取训练和测试的交集特征
        if data_config_test:
            csv_dir_test = data_config_test.pop('csv_dir')
            modality_test = pd.read_csv(os.path.join(csv_dir_test, 'Stats.csv'))
            Data_test = pd.read_csv(os.path.join(csv_dir_test, 'Data.csv'), index_col=0)
            if self.label2int:
                counts_test = Data_test['Label'].value_counts()
                self.print_log('Labels_test: ' + ', '.join([f'{kk}({self.label2int[kk]}):{counts_test[kk]}' for kk in counts_test.keys()]))
                Data_test['Label'] = Data_test['Label'].apply(lambda x: self.label2int.get(x))
                Data_test.index = Data_test.index.astype(str)
                Data_test = Data_test.dropna(subset=['Label'], axis=0, how='any')
            label_df_test = pd.DataFrame(dict(zip(Data_test.index, Data_test['Label'])), index=['true_label'])

            mod2var = dict()
            mod2var_test = dict()
            for g, c in modality.groupby(['Group']).groups.items():
                cols1 = modality.loc[c, 'Variable name'].tolist()
                cols2 = modality_test.loc[modality_test['Group'] == g, 'Variable name'].tolist()
                cols = [kk for kk in cols1 if kk in cols2]
                # print(cols1, cols2, cols)
                if len(cols) == 0:
                    continue
                idx1 = [list(Data.columns).index(kk) for kk in cols]
                idx2 = [list(Data_test.columns).index(kk) for kk in cols]
                mod2var[g] = [cols, idx1]
                mod2var_test[g] = [cols, idx2]
            self.print_log("Train data: " + utils.print_format({cc: mod2var[cc][0] for cc in mod2var.keys()}, form='%s'))
            self.print_log("Test data: " + utils.print_format({cc: mod2var_test[cc][0] for cc in mod2var_test.keys()}, form='%s'))

        else:
            modality = pd.read_csv(os.path.join(csv_dir, 'Stats.csv'))
            mod2var = dict()
            for g, c in modality.groupby(['Group']).groups.items():
                cols = modality.loc[c, 'Variable name'].tolist()
                idx = [list(Data.columns).index(kk) for kk in cols]
                mod2var[g] = [cols, idx]

        env_cols = data_config.pop('env')
        if env_cols:
            mod2var['Env'] = [env_cols, [list(Data.columns).index(kk) for kk in env_cols]]
            assert (self.config['algorithm']['fuse'] is None and
                    self.config['algorithm']['loss'] is not None and
                    (self.config['algorithm']['loss']['cf_coeff'] > 0 or
                     self.config['algorithm']['loss']['dist_coeff'] > 0))
            self.print_log(utils.print_format({cc: len(Data[cc].unique()) for cc in env_cols}, form='%d'))
        else:
            if self.config['algorithm']['loss'] is not None:
                assert (self.config['algorithm']['loss']['cf_coeff'] == 0 and
                        self.config['algorithm']['loss']['dist_coeff'] == 0)

        datasets = dict()
        feeder = self.config.pop('feeder')
        Feeder = utils.import_class(feeder)
        Prep = utils.import_class(feeder.split('.Feeder')[0] + '.preprocess')
        pre_config = {kk: self.config[kk] for kk in ['impute', 'normalize']}
        prox = 'MRIValue' if 'MRIValue' in mod2var.keys() else 'Gene'

        if split_package is None:
            pass

        else:  # 无模型，训练
            with open(os.path.join(csv_dir, split_package, 'cross5CV.pkl'), 'rb') as f:
                foldsID = pickle.load(f)
            foldsID = {kk: [str(ss) for ss in vv] for kk, vv in foldsID.items()}
            if len(data_config) == 0:
                Feeder0 = utils.import_class(feeder + 'CSV')
                for subset in ['train', 'val', 'test']:
                    data0 = Feeder0(data_config=data_config.copy(),
                                    raw=[self.label_df, foldsID], fold=self.config['fold'], subset=subset)
                    data0.data = Data.loc[data0.subject, mod2var[prox][0]].astype(np.float32)
                    data0.dims = data0.data.shape[1]
                    datasets[f'MRI_{subset}'] = data0
                data = {subset: datasets[f'MRI_{subset}'].data for subset in ['train', 'val', 'test']}
                if data_config_test:
                    data0 = Feeder0(data_config=data_config_test.copy(), raw=[label_df_test, None])
                    data0.data = Data_test.loc[data0.subject, mod2var_test[prox][0]].astype(np.float32)
                    data0.dims = data0.data.shape[1]
                    datasets['MRI_indi'] = data0
                    data['indi'] = data0.data
                pre = Prep(data, **pre_config)
                for kk in datasets.keys():
                    datasets[kk].data = pre[kk.split('MRI_')[1]]

            else:
                Feeder0 = utils.select_feeder(data_config, feeder)
                datasets.update({
                    f'MRI_{subset}': Feeder0(data_config=data_config.copy(),
                                             raw=[self.label_df, foldsID], fold=self.config['fold'], subset=subset)
                    for subset in ['train', 'val', 'test']
                })
                if data_config_test:
                    datasets['MRI_indi'] = Feeder0(data_config=data_config_test.copy(), raw=[label_df_test, None])
            image_dims = datasets['MRI_train'].dims

            for mm, cols in mod2var.items():
                if mm == prox:
                    continue
                data = {subset: Data.loc[datasets[f'MRI_{subset}'].subject, cols[0]] for subset in ['train', 'val', 'test']}
                if data_config_test:
                    data['indi'] = Data_test.loc[datasets['MRI_indi'].subject, cols[0]]
                if mm == 'Env' and 'OH.' in feeder:
                    onehot = utils.import_class(feeder.split('.Feeder')[0] + '.OneHot')
                    pre, new_col = onehot(data)
                    mod2var[mm][0] = new_col
                else:
                    pre = Prep(data, **pre_config)
                datasets.update({
                    f'{mm}_{subset}': Feeder(pre[subset]) for subset in ['train', 'val', 'test']
                })
                if data_config_test:
                    datasets[f'{mm}_indi'] = Feeder(pre['indi'])
            self.print_log('Sample number:\t' + utils.print_format(
                {subset: len(datasets[f'MRI_{subset}']) for subset in
                 ['train', 'val', 'test'] + (['indi'] if data_config_test else [])}, form='%d'))

        # loader_config = {'batch_size': self.config['batch_size'], 'num_workers': self.config['num_workers']}
        loader_config = {'batch_size': self.config['batch_size'], 'num_workers': 0}
        loader = dict()
        if self.operator == 'train':
            for name, dataset in datasets.items():
                if name.split('_')[1] == 'train':
                    loader[name] = DataLoader(dataset=dataset, shuffle=True, drop_last=True, **loader_config)
                elif name.split('_')[1] in ['val', 'test', 'indi']:
                    loader[name] = DataLoader(dataset=dataset, shuffle=False, drop_last=False, **loader_config)
        else:
            for name, dataset in datasets.items():
                loader[name] = DataLoader(dataset=dataset, shuffle=False, drop_last=False, **loader_config)
        return loader, mod2var, image_dims

    def cast_data(self, subset, image, label, index):
        data_dict = dict()
        for kk, vv in self.data_loader.items():
            if kk == f'MRI_{subset}':
                data_dict['MRI'] = image.to(self.device)
            elif kk.split('_')[1] == subset:
                data = vv.dataset[index]
                if len(data.shape) == 1:
                    data = np.expand_dims(data, axis=0)
                data_dict[kk.split('_')[0]] = torch.tensor(data).to(self.device)
        if self.num_class == 1:
            label = label.float().to(self.device)
        else:
            label = label.long().to(self.device)
        return data_dict, label

    @timeit
    def train_an_epoch(self, epoch):
        self.print_log(f'Train, epoch={epoch} ' + '-' * 10)
        self.tb_writer.add_scalar('epoch', epoch, self.global_step)
        loss_value = []
        stats = ['data', 'model', 'evaluate']
        log_time = dict(zip(stats, [1e-5] * len(stats)))
        # scores, true = [], []

        process = tqdm(self.data_loader['MRI_train'], ncols=50)
        try:
            # for batch_idx, (_, label, index) in enumerate(process):
            process_iter = iter(process)
            for i in range(len(self.data_loader['MRI_train'])):
                image, label, index = next(process_iter)
                self.global_step += 1

                cur_time = time.time()
                data, label = self.cast_data('train', image, label, index)
                log_time['data'] += time.time() - cur_time

                y_pred, losses = self.algorithm.update(data, label, epoch, log_time=log_time)
                # self.print_log(f"iter={i}, " + utils.print_format(losses, form="%.4f"))
                log_time['model'] += log_time['update']

                cur_time = time.time()
                loss_value.append(losses['Fuse_total'])
                for loss_name, loss in losses.items():
                    modal, loss_name = loss_name.split('_', 1)
                    self.tb_writer.add_scalar(f'{loss_name}_train/{modal}', loss, self.global_step)

                # true.extend(label.data.cpu().numpy())
                # scores.extend(y_pred.view(-1).data.cpu().numpy())
                value, predict_label = torch.max(y_pred.data, 1)
                acc = torch.mean((predict_label == label.data).float())
                self.tb_writer.add_scalar('acc/train', acc, self.global_step)
                log_time['evaluate'] += time.time() - cur_time

                try:
                    assert not np.isnan(losses['Fuse_total'])
                except AssertionError:
                    self.print_log('Loss value is NaN, stop training')
                    raise ValueError

        except KeyboardInterrupt:
            process.close()
            raise
        process.close()

        # proportion = {k: '{:02d}%'.format(int(round(log_time[k] * 100 / sum(log_time.values())))) for k in stats}
        # self.print_log('Time consumption: ' + utils.print_format(proportion, stats, '%s'))
        # # self.print_log('Time consumption: ' + utils.print_format(log_time, stats))
        self.print_log(utils.print_format(log_time))
        self.print_log(f'total_loss={np.mean(loss_value)}')

    @timeit
    def valid_an_epoch(self, epoch, subset='val'):
        self.print_log(f'Valid-{subset}, epoch={epoch} ' + '-' * 10)
        dataset = self.data_loader[f'MRI_{subset}'].dataset
        labels_dict = dict(zip(dataset.subject, dataset.label))
        score_dict = dict([])
        losses_value = dict()
        count = 0
        stats = ['data', 'model', 'evaluate']
        log_time = dict(zip(stats, [1e-5] * len(stats)))

        process = tqdm(self.data_loader[f'MRI_{subset}'], ncols=50)
        try:
            # for batch_idx, (_, label, index) in enumerate(process):
            process_iter = iter(process)
            for _ in range(len(self.data_loader[f'MRI_{subset}'])):
                image, label, index = next(process_iter)

                cur_time = time.time()
                data, label = self.cast_data(subset, image, label, index)
                log_time['data'] += time.time() - cur_time

                y_pred, losses = self.algorithm.validate(data, label, epoch, log_time=log_time)
                log_time['model'] += log_time['validate']

                cur_time = time.time()
                sub_list = dataset.subject[index] if len(index) > 1 else [dataset.subject[index]]
                score_dict.update(dict(zip(sub_list, y_pred.data.cpu().numpy())))  # 每个sub_name对应一个预测
                for loss_name, loss in losses.items():
                    if isinstance(loss, torch.Tensor):
                        loss = loss.item()
                    if loss_name not in losses_value.keys():
                        losses_value[loss_name] = loss * len(index)
                    else:
                        losses_value[loss_name] += loss * len(index)
                log_time['evaluate'] += time.time() - cur_time

                count += len(index)

        except KeyboardInterrupt:
            process.close()
            raise
        process.close()

        # assert (np.array(list(score_dict.keys())) == dataset.subject).all(), 'not match'
        labels = np.array([labels_dict[kk] for kk in score_dict.keys()])
        accuracy = np.mean(np.array(list(score_dict.values())).argmax(axis=1) == labels)

        if subset == 'val':
            self.algorithm.set_scheduler_step(losses=losses_value, acc=accuracy)
            for loss_name, loss in losses_value.items():
                loss = loss / count
                losses_value[loss_name] = loss
                modal, loss_name = loss_name.split('_', 1)
                self.tb_writer.add_scalar(f'{loss_name}_{subset}/{modal}', loss, self.global_step)
            for key, value in self.algorithm.lr.items():
                self.tb_writer.add_scalar('lr/'+key, value, self.global_step)
            self.tb_writer.add_scalar(f'acc/{subset}', accuracy, self.global_step)

            if accuracy > self.best_metric:
                self.best_metric = accuracy
                self.best_epoch = epoch
                self.algorithm.save_model(path=self.epoch_dir, epoch=epoch, flag='best')
                self.print_log(f'Save epoch{epoch}_best_model.pth')

        self.print_log(f"loss={losses_value['Fuse_total']}, acc={accuracy} (best={self.best_metric})")
        with open(os.path.join(self.epoch_dir, f'epoch{epoch}_{subset}_score.pkl'), 'wb') as f:
            pickle.dump(score_dict, f)

    @timeit
    def settle_results(self, epochs, rest):
        self.print_log('Settle results.' + '-' * 10)
        if self.num_class >= 2:
            metrics = ['acc', 'pre', 'sen', 'spe', 'f1', 'auc']
        elif self.num_class == 1:
            metrics = ['mae', 'mse', 'rmse', 'r2', 'Pear', 'corr']
        else:
            raise ValueError(f"Cannot assign metrics")
        ave = 'macro' if self.num_class >= 3 else None

        ss = SettleResults(self.label_df, self.config['task'], self.work_dir, metrics=metrics, ave=ave)
        ss.merge_pkl(num_epoch=epochs+1, subset='val')
        ss.concat_trend_scores(num_epoch=epochs+1, subset='val')
        for subset in set(rest) - {'val'}:
            ss.merge_pkl(num_epoch=epochs+1, start_epoch=epochs, subset=subset)
            ss.concat_trend_scores(num_epoch=epochs+1, start_epoch=epochs, subset=subset)
        # for subset in ['train', 'val', 'test']:
        #     ss.confusion_matrix(subset=subset, out_path=os.path.join(self.work_dir, f'CM_{subset}.png'))

    def post_evaluate(self, phase):
        self.print_log(f'Post-evaluate, {phase} ' + '-' * 10)
        subset = phase.split('_')[-1]
        dataset = self.data_loader[f'MRI_{subset}'].dataset
        labels_dict = dict(zip(dataset.subject, dataset.label))
        score_dict = dict([])

        process = tqdm(self.data_loader[f'MRI_{subset}'], ncols=50)
        try:
            # for batch_idx, (_, label, index) in enumerate(process):
            process_iter = iter(process)
            for _ in range(len(self.data_loader[f'MRI_{subset}'])):
                image, label, index = next(process_iter)
                data, label = self.cast_data(subset, image, label, index)
                mri = data['MRI']
                non = torch.cat([data[mm] for mm in self.algorithm.groups_nonMRI], dim=-1)
                y_pred = self.algorithm.post_evaluate(mri, non)

                sub_list = dataset.subject[index] if len(index) > 1 else [dataset.subject[index]]
                score_dict.update(dict(zip(sub_list, y_pred.data.cpu().numpy())))  # 每个sub_name对应一个预测

        except KeyboardInterrupt:
            process.close()
            raise
        process.close()

        # assert (np.array(list(score_dict.keys())) == dataset.subject).all(), 'not match'
        labels = np.array([labels_dict[kk] for kk in score_dict.keys()])
        accuracy = np.mean(np.array(list(score_dict.values())).argmax(axis=1) == labels)
        self.print_log(f"acc={accuracy}")

        if phase.startswith('allepo_'):
            epoch = int(phase.split('_')[1].split('.')[0])
            score_dict = {epoch: score_dict}
        with open(os.path.join(self.epoch_dir, f'{phase}_score.pkl'), 'wb') as f:
            pickle.dump(score_dict, f)

    def post_cam(self, phase):
        self.print_log(f'Post-cam, {phase} ' + '-' * 10)

        if phase.split('.')[-1] == 'CAM':
            cam_list = [
                'GradCAM', 'GradCAMPlusPlus', 'XGradCAM', 'LayerCAM',
                # 'ScoreCAM',  # 内存不够
                # 'EigenCAM', 'EigenGradCAM',  # 耗时太长
            ]
        else:
            cam_list = [phase.split('.')[-1]]

        for cam_name in cam_list:
            cam = utils.import_class('tools.grad_cam.' + cam_name)
            model = self.algorithm.all_networks['MRI']
            cam = cam(model=model,
                      target_layers=[model.encoder.networks[f'layer{kk}'].conv for kk in range(model.encoder.depth)])
            # cam = cam(model=model,
            #           target_layers=[model.encoder.networks['layer0'].conv])

            for subset in ['train', 'val', 'test'] + (['indi'] if 'MRI_indi' in self.data_loader.keys() else []):
                data_loader = self.data_loader[f'MRI_{subset}']
                dataset = data_loader.dataset
                score_dict1, score_dict2 = dict([]), dict([])
                process = tqdm(data_loader, ncols=50)
                try:
                    # for batch_idx, (_, label, index) in enumerate(process):
                    process_iter = iter(process)
                    for _ in range(len(data_loader)):
                        image, label, index = next(process_iter)
                        data, label = self.cast_data(subset, image, label, index)

                        grad_cam1 = cam(data['MRI'])

                        sub_list = dataset.subject[index] if len(index) > 1 else [dataset.subject[index]]
                        score_dict1.update(dict(zip(sub_list, grad_cam1)))

                except KeyboardInterrupt:
                    process.close()
                    raise
                process.close()

                with open(os.path.join(self.epoch_dir, f"{phase.rsplit('.', 1)[0]}.{cam_name}_{subset}.pkl"), 'wb') as f:
                    pickle.dump(score_dict1, f)

    def post_shap(self, phase):
        self.print_log(f'Post-shap, {phase} ' + '-' * 10)
        
        # Randomly select k samples for each class to form a background (k * num_class samples in total)
        # Because 'split' has controlled the train/val/test, 'seed' is used to control the random selection
        data_two = dict()
        try:
            data_two['MRI'] = torch.tensor(np.stack(self.data_loader[f'MRI_train'].dataset.image, axis=0)).unsqueeze(1)
        except AttributeError:
            data_two['MRI'] = torch.tensor(self.data_loader[f'MRI_train'].dataset.data)
        for kk, vv in self.data_loader.items():
            if kk.split('_')[1] == 'train' and kk != 'MRI_train':
                data = vv.dataset.data
                if len(data.shape) == 1:
                    data = np.expand_dims(data, axis=0)
                data_two[kk.split('_')[0]] = torch.tensor(data)
        slc_idx = utils.random_select(self.data_loader[f'MRI_train'].dataset.label, int(phase.split('_')[-1]))
        data_two = [
            data_two['MRI'][slc_idx].to(self.device),
            torch.cat([data_two[mm][slc_idx] for mm in self.algorithm.groups_nonMRI], dim=-1).to(self.device)
        ]
        e_two = shap.DeepExplainer(self.algorithm, data_two)
        del data_two

        self.print_log('Finish loading DeepExplainer')
        for subset in ['train', 'val', 'test'] + (['indi'] if 'MRI_indi' in self.data_loader.keys() else []):
            if os.path.isfile(os.path.join(self.epoch_dir, f"{phase}_{subset}.NonMRI.pkl")):
                continue
            data_loader = self.data_loader[f'MRI_{subset}']
            score_dict1, score_dict2 = dict([]), dict([])
            process = tqdm(data_loader, ncols=50)
            try:
                process_iter = iter(process)
                for i in range(len(data_loader)):
                    image, label, index = next(process_iter)
                    data, label = self.cast_data(subset, image, label, index)
                    mri = data['MRI']
                    non = torch.cat([data[mm] for mm in self.algorithm.groups_nonMRI], dim=-1)
                    # print(mri.shape, non.shape)
                    # shap_mri = e_MRI.shap_values(mri)
                    # shap_non = e_NonMRI.shap_values(non)
                    res = e_two.shap_values([mri, non])  # [c=0,c=1,c=2], 每个再包含2个tensor，分别对应mri和non
                    # print(res[0][0].shape)
                    shap_mri = np.stack([res[cc][0].squeeze() for cc in range(self.num_class)], axis=-1)
                    shap_non = np.stack([res[cc][1] for cc in range(self.num_class)], axis=-1)
                    # print(shap_mri.shape, shap_non.shape)

                    sub_list = data_loader.dataset.subject[index] if len(index) > 1 else [data_loader.dataset.subject[index]]
                    score_dict1.update(dict(zip(sub_list, shap_mri)))
                    # print(len(score_dict1), list(score_dict1.values())[-1].shape)
                    score_dict2.update(dict(zip(sub_list, shap_non)))

                with open(os.path.join(self.epoch_dir, f"{phase}_{subset}.MRI.pkl"), 'wb') as f:
                    pickle.dump(score_dict1, f)
                with open(os.path.join(self.epoch_dir, f"{phase}_{subset}.NonMRI.pkl"), 'wb') as f:
                    pickle.dump(score_dict2, f)
                self.print_log(f'Finish {subset}')

            except KeyboardInterrupt:
                process.close()
                raise
            process.close()

    def post_emb(self, phase):
        self.print_log(f'Post-emb, {phase} ' + '-' * 10)

        saveC, saveZ = dict(), dict()
        for subset in ['train', 'val', 'test'] + (['indi'] if 'MRI_indi' in self.data_loader.keys() else []):
            data_loader = self.data_loader[f'MRI_{subset}']
            dataset = data_loader.dataset
            score_dict1, score_dict2, score_dict3 = dict(), dict(), dict()
            score_dict4, score_dict5, score_dict6 = dict(), dict(), dict()
            process = tqdm(data_loader, ncols=50)
            try:
                # for batch_idx, (_, label, index) in enumerate(process):
                process_iter = iter(process)
                for _ in range(len(data_loader)):
                    image, label, index = next(process_iter)
                    data, label = self.cast_data(subset, image, label, index)
                    mri = data['MRI']
                    non = torch.cat([data[mm] for mm in self.algorithm.groups_nonMRI], dim=-1)

                    emb_mri_c, emb_mri = self.algorithm.all_networks['MRI'].get_z(mri)  # 前一个zc，后一个z
                    emb_non_c, emb_non = self.algorithm.all_networks['NonMRI'].get_z(non)
                    emb = torch.cat([emb_mri, emb_non], dim=-1)
                    emb_fuse_c, emb_fuse = self.algorithm.all_networks['Fuse'].get_z(emb)

                    sub_list = dataset.subject[index] if len(index) > 1 else [dataset.subject[index]]
                    score_dict1.update(dict(zip(sub_list, emb_mri_c.data.cpu().numpy())))
                    score_dict2.update(dict(zip(sub_list, emb_non_c.data.cpu().numpy())))
                    score_dict4.update(dict(zip(sub_list, emb_mri.data.cpu().numpy())))
                    score_dict5.update(dict(zip(sub_list, emb_non.data.cpu().numpy())))
                    if emb_fuse_c is not None:
                        score_dict3.update(dict(zip(sub_list, emb_fuse_c.data.cpu().numpy())))
                        score_dict6.update(dict(zip(sub_list, emb_fuse.data.cpu().numpy())))

                    # if phase.endswith('emb'):
                    #     score_dict1.update(dict(zip(sub_list, emb_mri_c.data.cpu().numpy())))
                    #     score_dict2.update(dict(zip(sub_list, emb_non_c.data.cpu().numpy())))
                    #     score_dict3.update(dict(zip(sub_list, emb_fuse_c.data.cpu().numpy())))
                    # elif phase.endswith('embZ'):
                    #     score_dict1.update(dict(zip(sub_list, emb_mri.data.cpu().numpy())))
                    #     score_dict2.update(dict(zip(sub_list, emb_non.data.cpu().numpy())))
                    #     score_dict3.update(dict(zip(sub_list, emb_fuse.data.cpu().numpy())))

            except KeyboardInterrupt:
                process.close()
                raise
            process.close()

            saveC[subset] = dict(MRI=score_dict1, NonMRI=score_dict2, Fuse=score_dict3)
            saveZ[subset] = dict(MRI=score_dict4, NonMRI=score_dict5, Fuse=score_dict6)

        with open(os.path.join(self.epoch_dir, f"{phase}C.pkl"), 'wb') as f:
            pickle.dump(saveC, f)
        with open(os.path.join(self.epoch_dir, f"{phase}Z.pkl"), 'wb') as f:
            pickle.dump(saveZ, f)

    def print_log(self, string, print_time=True):
        if print_time:
            string = f"[{utils.local_time()}] {string}"
        print(string)
        if self.config['print_log']:
            with open(os.path.join(self.work_dir, 'log.txt'), 'a') as f:
                print(string, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./conf/config.json', type=str)
    parser.add_argument('--custom', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        default_config = json.load(f)
    if args.custom:
        print(args.custom)
        config = json.loads(args.custom)
    else:
        config = default_config

    # print(config)
    # parameter priority: argparse (string from command line) > config.json > DEFAULT_*
    config = utils.fill_default_config(config, default_config, flag='rootDeep')
    # print(config)
    processor = Processor(config)
    processor.main()
