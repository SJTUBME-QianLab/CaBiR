import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tools import utils
from tools.catch import CatchNetwork, CatchOptimizer, CatchScheduler
from tools.evaluation import get_evaluations, get_auc


class Algorithm(nn.Module):
    def __init__(self, num_class, hparams):
        super().__init__()
        self.hparams = hparams.copy()
        self.num_class = num_class
        self.sch_split = hparams.pop('sch_split')

        self.all_networks = nn.ModuleDict()
        self.all_optimizers = dict()
        self.all_schedulers = dict()
        self.base_loss = self.get_loss()
        self.lr = dict()

    def update(self, **kwargs):
        raise NotImplementedError

    def validate(self, **kwargs):
        raise NotImplementedError

    def calculate_loss(self, **kwargs):
        raise NotImplementedError

    def get_loss(self):
        if self.num_class == 1:
            loss = nn.MSELoss()
        elif self.num_class > 1:
            loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f'self.num_class={self.num_class}')
        return loss

    def save_model(self, path='./', epoch=0, flag=''):
        if flag == 'best':
            utils.remove_files(path, 'best_model.pth')
        save = {
            'epoch': epoch,
            'all_networks': {mm: model.state_dict() for mm, model in self.all_networks.items()},
        }
        if not (self.fuse_str is None or self.fuse_str == 'cat'):
            save.update({'modality_weights': self.all_networks['Fuse'].fuse_weight})
        torch.save(save, os.path.join(path, f'epoch{epoch}_{flag}_model.pth'))

    def load_model(self, path='./', select='final'):
        pth_path = utils.find_dir(path, f'{select}_model.pth')
        if len(pth_path) == 1:
            checkpoint = torch.load(pth_path[0])
            for mm, model in self.all_networks.items():
                model.load_state_dict(checkpoint['all_networks'][mm])
            if not (self.fuse_str is None or self.fuse_str == 'cat'):
                self.all_networks['Fuse'].fuse_weight = checkpoint['modality_weights']
        else:
            raise ValueError(f'cannot find {select}_model.pth in {path}')

    def set_train_status(self, status, model_list=None):
        if model_list is None:
            model_list = self.all_networks.values()
        for model in model_list:
            model.train(status)

    def set_zero_grad(self, optim_list=None):
        if optim_list is None:
            optim_list = self.all_optimizers.values()
        for optim in optim_list:
            optim.zero_grad()

    def set_optim_step(self, optim_list=None):
        if optim_list is None:
            optim_list = self.all_optimizers.values()
        for optim in optim_list:
            optim.step()

    def set_scheduler_step(self, **kwargs):
        def set_scheduler_step1(losses, acc, scheduler_dict=None):
            if scheduler_dict is None:
                scheduler_dict = self.all_schedulers
            assert self.hparams['scheduler']['name'] == 'ReduceLROnPlateau' and self.sch_mode == 'min'
            for key, scheduler in scheduler_dict.items():
                loss = losses[f'{key}_total']
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                scheduler.step(loss)
                self.lr[key] = self.all_optimizers[key].param_groups[0]['lr']

        def set_scheduler_step2(losses, acc, scheduler_dict=None):
            if scheduler_dict is None:
                scheduler_dict = self.all_schedulers
            loss = losses['Fuse_total'] if len(self.all_networks) == 3 \
                else losses['MRI_total'] if 'MRI_total' in losses.keys() \
                else losses['NonMRI_total']
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            for key, scheduler in scheduler_dict.items():
                if self.hparams['scheduler']['name'] == 'ReduceLROnPlateau':
                    if self.sch_mode == 'min':
                        scheduler.step(loss)
                    elif self.sch_mode == 'max':
                        scheduler.step(acc)
                else:
                    scheduler.step()
                self.lr[key] = self.all_optimizers[key].param_groups[0]['lr']

        return set_scheduler_step1(**kwargs) if self.sch_split else set_scheduler_step2(**kwargs)


class Algorithm1(Algorithm):
    def __init__(self, n_inputs, num_epochs, num_class, hparams):
        super().__init__(num_class, hparams)
        self.n_inputs = n_inputs
        self.num_epochs = num_epochs
        if 'nonMRI' in hparams:
            groups_nonMRI = hparams.pop('nonMRI')
        else:
            groups_nonMRI = ['Demo', 'Func', 'MedH', 'Psychia', 'Psycho']
        self.groups_nonMRI = [kk for kk in groups_nonMRI if kk in self.n_inputs.keys()]
        self.dims_in = {
            'Env': self.n_inputs['Env'] if 'Env' in self.n_inputs.keys() else 0,
            'MRI': self.n_inputs['MRI'],  # 155 or [182, 218, 182]
            'NonMRI': sum([self.n_inputs[mm] for mm in self.groups_nonMRI]),  # 74
        }

        config_model1, config_model2, config_model3, config_opt, config_sch, config_loss, config_ampl = [
            hparams.pop(kk) for kk in ['model1', 'model2', 'model3', 'optimizer', 'scheduler', 'loss', 'ampl']]
        self.base_lr = config_opt.pop('lr')
        self.fuse_str = hparams.pop('fuse')
        if config_ampl is not None:
            config_ampl['num_epoch'] = self.num_epochs
        if all([config_model1, config_model2, config_model3]):
            if not self.sch_split:
                assert config_model1['lr_c'] == config_model2['lr_c'] == config_model3['lr_c'] == 1
        else:
            assert self.fuse_str is None

        assert len(hparams) == 0, f'Invalid hparams: {hparams}'
        if config_sch['name'] == 'ReduceLROnPlateau':
            self.sch_mode = config_sch['mode']
        else:
            self.sch_mode = None

        # MRI
        if config_model1 is not None:
            config_model1['enc']['n_inputs'] = n_inputs['MRI']
            config_opt['lr'] = config_model1.pop('lr_c') * self.base_lr
            if config_ampl is None and config_loss is None:
                network = OneModality(self.num_class, config_model1.copy())
            else:
                network = OneModalityCausal(self.num_class, config_model1.copy(),
                                            config_loss, config_ampl)
            self.dims_in['MRI'] = network.dim
            self.all_networks.add_module('MRI', network)
            optimizer = CatchOptimizer(network, config_opt)
            self.all_optimizers['MRI'] = optimizer
            self.all_schedulers['MRI'] = CatchScheduler(optimizer, config_sch)

        # NonMRI
        if config_model2 is not None:
            config_model2['enc']['n_inputs'] = self.dims_in['NonMRI']
            config_opt['lr'] = config_model2.pop('lr_c') * self.base_lr
            if config_ampl is None and config_loss is None:
                network = OneModality(self.num_class, config_model2.copy())
            else:
                network = OneModalityCausal(self.num_class, config_model2.copy(),
                                            config_loss, config_ampl)
            self.all_networks.add_module('NonMRI', network)
            optimizer = CatchOptimizer(network, config_opt)
            self.all_optimizers['NonMRI'] = optimizer
            self.all_schedulers['NonMRI'] = CatchScheduler(optimizer, config_sch)

        # Multi-modal Fuse
        if all([config_model1, config_model2]):
            config_model3['enc']['n_inputs'] = sum(network.dim for network in self.all_networks.values())
            config_opt['lr'] = config_model3.pop('lr_c') * self.base_lr

            if self.fuse_str is None and config_ampl is not None and config_loss is not None:
                network = FuseCausal(self.num_class, config_model3.copy(),
                                     config_loss, config_ampl,
                                     dim_env=self.dims_in['Env'])
            elif self.fuse_str.split('_')[0] in ['train', 'ave', 'SMave', 'median']:
                assert self.dims_in['Env'] == 0 and len(config_model3['enc']) == 1 and "dim" not in config_model3.keys()
                network = Fuse(self.num_class, self.fuse_str)
            elif self.fuse_str == 'cat':
                assert self.dims_in['Env'] == 0
                network = FuseNet(self.num_class, config_model3.copy())
            else:
                raise ValueError(f'Invalid config: ampl: {config_ampl}, loss: {config_loss}, fuse_str:{self.fuse_str}')

            self.all_networks.add_module('Fuse', network)
            # print(self.all_networks)

            if self.fuse_str is None or self.fuse_str == 'cat':
                optimizer = CatchOptimizer(network, config_opt)
                self.all_optimizers['Fuse'] = optimizer
                self.all_schedulers['Fuse'] = CatchScheduler(optimizer, config_sch)

    def cat_data(self, data):
        data_2 = dict()
        data_2['MRI'] = data['MRI']
        data_2['NonMRI'] = torch.cat([data[mm] for mm in self.groups_nonMRI], dim=-1)
        if len(self.all_networks) == 3 and self.dims_in['Env'] > 0:
            assert 'Env' in data.keys()
            data_2['Env'] = data['Env']
        else:
            assert 'Env' not in data.keys()
        return data_2

    @utils.timeit
    def calculate_loss(self, data, label, epoch, log_time=None):
        data = self.cat_data(data)
        losses_items = dict()
        emb_split = dict()
        pred_split = dict()

        for mm, model in self.all_networks.items():
            if mm in ['Fuse']:
                continue
            modality = data[mm]
            emb_split[mm], pred_split[mm], losses_items[mm] = model.cal_loss(modality, label, epoch)
        losses_items = {f'{mm}_{kk}': ll[kk] for mm, ll in losses_items.items() for kk in ll.keys()}
        emb = torch.cat([emb_split[mm] for mm in emb_split.keys()], dim=-1)

        if len(self.all_networks) == 3:
            if self.fuse_str is None:
                assert hasattr(self.all_networks['Fuse'], 'dim_env')
                if self.dims_in['Env'] > 0:
                    pred, losses_items_ = self.all_networks['Fuse'].cal_loss(emb.detach(), label, data['Env'], epoch)
                else:
                    pred, losses_items_ = self.all_networks['Fuse'].cal_loss(emb.detach(), label, None, epoch)
            elif self.fuse_str == 'cat':
                _, pred, losses_items_ = self.all_networks['Fuse'].cal_loss(emb.detach(), label, epoch)
            elif self.fuse_str.split('_')[0] in ['train', 'ave', 'SMave', 'median']:
                pred, losses_items_ = self.all_networks['Fuse'].cal_loss(pred_split, label, epoch)
            else:
                raise ValueError(f'Invalid fuse_str: {self.fuse_str}')
            losses_items.update({f'Fuse_{kk}': ll for kk, ll in losses_items_.items()})
        else:
            assert len(self.all_networks) == 1
            pred = list(pred_split.values())[0]
            losses_items['Fuse_total'] = losses_items['MRI_total'] if 'MRI_total' in losses_items.keys() \
                else losses_items['NonMRI_total']
        return pred, losses_items

    @utils.timeit
    def validate(self, data, label, epoch, log_time=None):
        self.set_train_status(False, self.all_networks.values())
        # self.eval()
        return self.calculate_loss(data, label, epoch, log_time=log_time)

    def post_evaluate(self, data_mri, data_nonmri):
        self.set_train_status(False, self.all_networks.values())
        # self.eval()
        return self.forward(data_mri, data_nonmri)

    @utils.timeit
    def update(self, data, label, epoch, log_time=None):
        self.set_train_status(True, self.all_networks.values())
        # self.train()

        pred, losses_items = self.calculate_loss(data, label, epoch, log_time=log_time)

        for mm in self.all_optimizers.keys():
            self.set_zero_grad([self.all_optimizers[mm]])
            losses_items[f'{mm}_total'].backward()
            self.set_optim_step([self.all_optimizers[mm]])
        losses_items['Fuse_total'] = losses_items['Fuse_total'].item()

        self.set_train_status(False, self.all_networks.values())
        # self.eval()

        return pred, losses_items
    
    def forward(self, data_mri, data_nonmri):
        if len(self.all_networks) == 3:
            emb_mri, pred_mri = self.all_networks['MRI'].get_z_pred(data_mri)
            emb_nonmri, pred_nonmri = self.all_networks['NonMRI'].get_z_pred(data_nonmri)
            emb = torch.cat([emb_mri, emb_nonmri], dim=-1)
            if self.fuse_str is None or self.fuse_str == 'cat':
                pred = self.all_networks['Fuse'](emb)
            else:
                pred = self.all_networks['Fuse'](pred_mri, pred_nonmri)
        else:
            assert len(self.all_networks) == 1
            if 'MRI' in self.all_networks.keys():
                pred = self.all_networks['MRI'](data_mri)
            else:
                pred = self.all_networks['NonMRI'](data_nonmri)
        return pred


class OneModality(nn.Module):
    def __init__(self, n_class, config_model):
        super().__init__()

        self.n_class = n_class
        self.CE = nn.CrossEntropyLoss(reduction='none')
        assert 'bias' not in config_model.keys()

        if 'dim' in config_model.keys():  # MLP
            self.dim = config_model.pop('dim')
            config_model['enc']['n_outputs'] = self.dim
            self.encoder = CatchNetwork(config_model.pop('enc'))
        elif config_model['enc']['name'].startswith('GNN'):
            self.encoder = CatchNetwork(config_model.pop('enc'))
            self.dim = self.encoder.dim
        else:  # CNN
            image_dim = config_model['enc'].pop('n_inputs')
            self.encoder = CatchNetwork(config_model.pop('enc'))
            self.dim = self.encoder.size(in_dims=image_dim)
        assert len(config_model) == 0
        self.fc = nn.Linear(self.dim, self.n_class)

    def cal_loss(self, x, y, epoch):
        z = self.encoder(x)
        pred = self.fc(z)
        loss = self.CE(pred, y)
        if hasattr(self.encoder, 'loss'):
            loss1 = self.encoder.loss()
            res = {'total': loss.mean() + loss1}
        else:
            res = {'total': loss.mean()}
        return z, pred, res

    def get_z(self, x):
        z = self.encoder(x)
        return z, z

    def get_z_pred(self, x):
        z = self.encoder(x)
        pred = self.fc(z)
        return z, pred

    def forward(self, x):
        z = self.encoder(x)
        pred = self.fc(z)
        return pred


class OneModalityCausal(nn.Module):
    def __init__(self, n_class, config_model, config_loss, config_ampl):
        super().__init__()

        self.n_class = n_class
        self.CE = nn.CrossEntropyLoss(reduction='none')
        bias = config_model.pop('bias')

        if 'dim' in config_model.keys():  # MLP
            self.dim = config_model.pop('dim')
            config_model['enc']['n_outputs'] = self.dim
            self.encoder = CatchNetwork(config_model.pop('enc'))
        elif config_model['enc']['name'] == 'BrainGNN':
            self.encoder = CatchNetwork(config_model.pop('enc'))
            self.dim = self.encoder.dim3
        elif config_model['enc']['name'].startswith('GNN'):
            self.encoder = CatchNetwork(config_model.pop('enc'))
            self.dim = self.encoder.dim
        else:  # CNN
            image_dim = config_model['enc'].pop('n_inputs')
            self.encoder = CatchNetwork(config_model.pop('enc'))
            self.dim = self.encoder.size(in_dims=image_dim)
        assert len(config_model) == 0

        self.encoder_bias = CatchNetwork({**bias, 'n_inputs': self.dim, 'n_outputs': self.dim})
        self.fc_causal = nn.Linear(self.dim, self.n_class)
        self.fc_bias = nn.Linear(self.dim, self.n_class)

        config_loss = config_loss.copy()
        self.GCE = GeneralizedCELoss(q=config_loss.pop('GCE_q'))
        self.ampl_type = config_loss.pop('ampl_type')  # perm, comb, none
        self.weight_type = config_loss.pop('weight_type')  # ratio, sig, none

        # amplitude
        self.lambda_ampl = Amplification(config_ampl.copy(), ampl_type=self.ampl_type)

    def block(self, z, y, y_g):
        pred_c = self.fc_causal(z)
        loss_CE_c = self.CE(pred_c, y)

        pred_b = self.fc_bias(z)
        loss_CE_b = self.CE(pred_b, y)
        loss_GCE_b = self.GCE(pred_b, y_g)
        return pred_c, loss_CE_c, loss_CE_b, loss_GCE_b

    def get_weight(self, loss_c, loss_b):
        loss_c = loss_c.clone().detach()
        loss_b = loss_b.clone().detach()
        if self.weight_type == 'zero':
            return (torch.ones_like(loss_c) * 0.5).to(loss_c.device)
        if self.weight_type == 'ratio':
            weight = loss_b / (loss_c + loss_b + 1e-8)  # (0,1)
        elif self.weight_type == 'sig':
            weight = torch.sigmoid(loss_b - loss_c)  # (0,1)
        else:
            raise ValueError(f'Invalid weight_type: {self.weight_type}')
        assert weight.requires_grad is False
        return weight

    def permutation(self, z_c, z_b, y):
        indices = np.random.permutation(len(y))
        z_perm = z_c + z_b[indices]
        y_perm = y[indices]  # label of z_b
        _, loss_CE_pc, loss_CE_pb, loss_GCE_pb = self.block(z_perm, y, y_perm)
        return loss_CE_pc, loss_CE_pb, loss_GCE_pb

    def combination(self, z_c, z_b, y):
        loss_CE_pc = torch.tensor([]).to(y.device)
        loss_CE_pb = torch.tensor([]).to(y.device)
        loss_GCE_pb = torch.tensor([]).to(y.device)
        for i, zi in enumerate(z_c):
            z_perm_i = zi.repeat(len(y), 1) + z_b
            y_c = y[i].repeat(len(y))
            _, loss_CE_pc_i, loss_CE_pb_i, loss_GCE_pb_i = self.block(z_perm_i, y_c, y)
            loss_CE_pc = torch.cat([loss_CE_pc, loss_CE_pc_i.unsqueeze(0)], dim=0)
            loss_CE_pb = torch.cat([loss_CE_pb, loss_CE_pb_i.unsqueeze(0)], dim=0)
            loss_GCE_pb = torch.cat([loss_GCE_pb, loss_GCE_pb_i.unsqueeze(0)], dim=0)
        return loss_CE_pc, loss_CE_pb, loss_GCE_pb

    def cal_loss(self, x, y, epoch=0):
        z = self.encoder(x)
        pred_c, loss_CE_c, loss_CE_b, loss_GCE_b = self.block(z, y, y)

        weight_main = self.get_weight(loss_CE_c, loss_CE_b)
        loss_main = weight_main * loss_CE_c + (1 - weight_main) * loss_GCE_b
        loss_total = loss_main.mean()
        res = {
            'main': loss_main.mean().item(),
            'weight_main': weight_main.mean().item(),
            'CE_c': loss_CE_c.mean().item(),
            'CE_b': loss_CE_b.mean().item(),
            'GCE_b': loss_GCE_b.mean().item(),
        }

        z_b = self.encoder_bias(z)
        z_c = z - z_b.detach()

        if self.ampl_type == 'comb':
            loss_CE_pc, loss_CE_pb, loss_GCE_pb = self.combination(z_c, z_b, y)
        elif self.ampl_type == 'perm':
            loss_CE_pc, loss_CE_pb, loss_GCE_pb = self.permutation(z_c, z_b, y)
        else:
            raise ValueError(f'Invalid ampl_type: {self.ampl_type}')
        res.update({
            'CE_pc': loss_CE_pc.mean().item(),
            'CE_pb': loss_CE_pb.mean().item(),
            'GCE_pb': loss_GCE_pb.mean().item(),
        })

        losses_items = dict(loss_CE_c=loss_CE_c, loss_CE_b=loss_CE_b, loss_CE_pc=loss_CE_pc, loss_CE_pb=loss_CE_pb)
        amplitude = self.lambda_ampl(epoch, losses_items)

        if amplitude > 0:
            weight_amp = self.get_weight(loss_CE_pc, loss_CE_pb)
            loss_amp = weight_amp * loss_CE_pc + (1 - weight_amp) * loss_GCE_pb
            loss_total += amplitude * loss_amp.mean()
            res.update({
                'total': loss_total,
                'amplitude': amplitude,
                'amp': loss_amp.mean().item(),
                'weight_amp': weight_amp.mean().item(),
            })
        else:
            res.update({
                'total': loss_total,
                'amplitude': 0,
                'amp': 0,
                'weight_amp': 0,
            })
        return z_c, pred_c, res

    def get_z(self, x):
        z = self.encoder(x)
        z_b = self.encoder_bias(z)
        z_c = z - z_b.detach()
        return z_c, z

    def get_z_pred(self, x):
        z = self.encoder(x)
        pred_c = self.fc_causal(z)
        z_b = self.encoder_bias(z)
        z_c = z - z_b.detach()
        return z_c, pred_c

    def forward(self, x):
        z = self.encoder(x)
        pred_c = self.fc_causal(z)
        return pred_c


class Fuse(nn.Module):
    def __init__(self, n_class, fuse_str):
        super().__init__()
        self.n_class = n_class
        self.fuse, self.metric_name, self.ave = utils.parse_fuse(fuse_str, n_class)
        self.fuse_weight = None
        self.CE = nn.CrossEntropyLoss(reduction='none')

    def predict(self, pred_split):
        if self.fuse == 'train':
            pred_all = torch.stack([pred_split[mm] * self.fuse_weight[mm] for mm in pred_split.keys()], dim=-1)
            pred = torch.sum(pred_all, dim=-1)
        else:
            pred_all = torch.stack(list(pred_split.values()), dim=-1)
            if self.fuse == 'ave':
                pred = torch.mean(pred_all, dim=-1)
            elif self.fuse == 'SMave':
                pred_all = F.softmax(pred_all, dim=-2)
                pred = torch.mean(pred_all, dim=-1)
            elif self.fuse == 'median':
                pred = torch.median(pred_all, dim=-1)
            else:
                raise ValueError(f'Fuse method not supported: {self.fuse}')

        return pred

    def fusion(self, pred_split, label):
        metric = dict()
        true = pd.DataFrame([label.data.cpu().numpy()], index=['true'])
        for mm, pred in pred_split.items():
            pred = pd.DataFrame(pred.data.cpu().numpy().T, index=[f'pred{k}' for k in range(self.n_class)])
            scores = pd.concat([pred, true], axis=0)
            if self.metric_name == 'auc':
                metric[mm], _, _ = get_auc(scores=scores, num_class=self.n_class, ave=self.ave)
            elif self.metric_name in utils.metric_set['clf'] + utils.metric_set['reg']:
                metric[mm] = get_evaluations(scores=scores, num_class=self.n_class, metrics=[self.metric_name], ave=self.ave)[0]
            else:
                raise ValueError(f'Metric not supported: {self.metric_name}')

        weights = {kk: np.exp(dd) for kk, dd in metric.items()}
        weights_sum = sum(weights.values())
        return {kk: dd / weights_sum for kk, dd in weights.items()}

    def cal_loss(self, pred_split, label, epoch=0):
        if self.fuse == 'train':
            self.fuse_weight = self.fusion(pred_split, label)
        pred = self.predict(pred_split)
        losses_items = {
            'total': self.CE(pred, label).mean(),
        }
        return pred, losses_items

    def get_z(self, x):
        return None, None

    def forward(self, pred_mri, pred_nonmri):
        pred = self.predict({'MRI': pred_mri, 'NonMRI': pred_nonmri})
        return pred


class FuseNet(OneModality):
    def __init__(self, n_class, config_model):
        nn.Module.__init__(self)

        self.n_class = n_class
        self.CE = nn.CrossEntropyLoss(reduction='none')
        assert 'bias' not in config_model.keys() and 'env' not in config_model.keys()

        if 'dim' in config_model.keys():
            self.dim = config_model.pop('dim')
            config_model['enc']['n_outputs'] = self.dim
            self.encoder = CatchNetwork(config_model.pop('enc'))
        else:
            assert 'name' not in config_model['enc']
            self.dim = config_model['enc']['n_inputs']
            config_model.pop('enc')
            self.encoder = lambda x: x
        assert len(config_model) == 0

        self.fc = nn.Linear(self.dim, self.n_class)


class FuseCausal(nn.Module):
    def __init__(self, n_class, config_model, config_loss, config_ampl, dim_env=0):
        super().__init__()

        self.n_class = n_class
        self.CE = nn.CrossEntropyLoss(reduction='none')
        bias = config_model.pop('bias')
        self.dim_env = dim_env

        if 'dim' in config_model.keys():
            self.dim = config_model.pop('dim')
            config_model['enc']['n_outputs'] = self.dim
            self.encoder = CatchNetwork(config_model.pop('enc'))
        else:
            assert 'name' not in config_model['enc']
            self.dim = config_model['enc']['n_inputs']
            config_model.pop('enc')
            self.encoder = lambda x: x

        self.encoder_bias = CatchNetwork({**bias, 'n_inputs': self.dim, 'n_outputs': self.dim})
        self.fc_causal = nn.Linear(self.dim, self.n_class)
        self.fc_bias = nn.Linear(self.dim, self.n_class)
        self.fc_final = nn.Linear(self.dim, self.n_class)

        config_loss = config_loss.copy()
        self.GCE = GeneralizedCELoss(q=config_loss.pop('GCE_q'))
        self.ampl_type = config_loss.pop('ampl_type')  # perm, comb, none
        self.weight_type = config_loss.pop('weight_type')  # ratio, sig, none
        self.cf_coeff = config_loss.pop('cf_coeff')
        if self.cf_coeff > 0:
            env = config_model.pop('env')
            assert self.dim_env > 0
            self.encoder_env = CatchNetwork({**env, 'n_inputs': self.dim_env, 'n_outputs': self.dim})
        else:
            assert self.dim_env == 0
        if self.cf_coeff > 0:
            self.fc_counterfactual = nn.Linear(self.dim, self.n_class)

        assert len(config_loss) == 0, f'Invalid config_loss: {config_loss}'
        assert len(config_model) == 0

        # amplitude
        self.lambda_ampl = Amplification(config_ampl.copy(), ampl_type=self.ampl_type)

    def block(self, z, y, y_g):
        pred_c = self.fc_causal(z)
        loss_CE_c = self.CE(pred_c, y)

        pred_b = self.fc_bias(z)
        loss_CE_b = self.CE(pred_b, y)
        loss_GCE_b = self.GCE(pred_b, y_g)
        return pred_c, loss_CE_c, loss_CE_b, loss_GCE_b

    def get_weight(self, loss_c, loss_b):
        loss_c = loss_c.clone().detach()
        loss_b = loss_b.clone().detach()
        if self.weight_type == 'zero':
            return (torch.ones_like(loss_c) * 0.5).to(loss_c.device)
        if self.weight_type == 'ratio':
            weight = loss_b / (loss_c + loss_b + 1e-8)  # (0,1)
        elif self.weight_type == 'sig':
            weight = torch.sigmoid(loss_b - loss_c)  # (0,1)
        else:
            raise ValueError(f'Invalid weight_type: {self.weight_type}')
        assert weight.requires_grad is False
        return weight

    def permutation(self, z_c, z_b, y):
        indices = np.random.permutation(len(y))
        z_perm = z_c + z_b[indices]
        y_perm = y[indices]  # label of z_b
        _, loss_CE_pc, loss_CE_pb, loss_GCE_pb = self.block(z_perm, y, y_perm)
        return loss_CE_pc, loss_CE_pb, loss_GCE_pb

    def combination(self, z_c, z_b, y):
        loss_CE_pc = torch.tensor([]).to(y.device)
        loss_CE_pb = torch.tensor([]).to(y.device)
        loss_GCE_pb = torch.tensor([]).to(y.device)
        for i, zi in enumerate(z_c):
            z_perm_i = zi.repeat(len(y), 1) + z_b
            y_c = y[i].repeat(len(y))
            _, loss_CE_pc_i, loss_CE_pb_i, loss_GCE_pb_i = self.block(z_perm_i, y_c, y)
            loss_CE_pc = torch.cat([loss_CE_pc, loss_CE_pc_i.unsqueeze(0)], dim=0)
            loss_CE_pb = torch.cat([loss_CE_pb, loss_CE_pb_i.unsqueeze(0)], dim=0)
            loss_GCE_pb = torch.cat([loss_GCE_pb, loss_GCE_pb_i.unsqueeze(0)], dim=0)
        return loss_CE_pc, loss_CE_pb, loss_GCE_pb

    def counterfactual(self, z_c, z_e, y):
        if self.cf_coeff == 0:
            return torch.tensor(0.).to(z_c.device)
        n_sample = len(z_c)
        z_cn = z_c.unsqueeze(0).repeat(n_sample, 1, 1)
        z_en = z_e.unsqueeze(0).repeat(n_sample, 1, 1)
        y_cn = y.unsqueeze(0).repeat(n_sample, 1)  # row same, column different, ==> z_e
        z_cf = z_cn.transpose(0, 1) + z_en
        z_cc = z_cn + z_en
        # z_cc = (z_c + z_e).repeat(n_sample, 1, 1)  # another way to get z_cc
        self.fc_counterfactual.weight.requires_grad = False
        self.fc_counterfactual.bias.requires_grad = False
        pred_cf = F.softmax(self.fc_counterfactual(z_cf), dim=-1)
        pred_cc = F.softmax(self.fc_counterfactual(z_cc), dim=-1)
        self.fc_counterfactual.weight.requires_grad = True
        self.fc_counterfactual.bias.requires_grad = True

        # y_cn convert to one-hot encoding, used to find the channel corresponding to GT class
        slc = F.one_hot(y_cn, num_classes=self.n_class).float()
        # only GT class channel
        eq_mask = y.unsqueeze(0) != y.unsqueeze(1)  # [n, n] Symmetric array, unequal class -- 1, otherwise 0
        # only GT class channel of sample pair with unequal classes
        slc = slc * eq_mask.unsqueeze(2).repeat(1, 1, self.n_class)  # [n, n, C]
        assert slc.shape == pred_cf.shape and (slc.max() == 1 or slc.max() == 0)
        # minimize part of pred_cf > pred_cc ;
        # Sum up with dim=-1 to eliminate classes channel; sum up with dim=0 for all samples;
        # divide by the number of samples with different labels for each sample (column)
        loss = (slc * F.relu(pred_cf - pred_cc)).sum(dim=-1).sum(dim=0) / (eq_mask.sum(dim=0) + 1e-8)
        ce_loss = self.CE(self.fc_counterfactual(z_c.detach() + z_e), y)
        # print(loss.shape, ce_loss.shape)
        # print('loss' + '-'*10 + '\n', loss.data.cpu().numpy())
        # print('ce_loss' + '-'*10 + '\n', ce_loss.data.cpu().numpy())
        return loss * self.cf_coeff + ce_loss
        # former item: only effect z_cf, z_cc, i.e., env_enc, enc, bias_enc; NOT effect fc_counterfactual
        # latter item: NOT effect z_c, i.e., enc, bias_enc; only effect fc_counterfactual, env_enc

    def cal_loss(self, x, y, env, epoch=0):
        z = self.encoder(x)
        pred_c, loss_CE_c, loss_CE_b, loss_GCE_b = self.block(z, y, y)

        weight_main = self.get_weight(loss_CE_c, loss_CE_b)
        loss_main = weight_main * loss_CE_c + (1 - weight_main) * loss_GCE_b
        loss_total = loss_main.mean()
        res = {
            'main': loss_main.mean().item(),
            'weight_main': weight_main.mean().item(),
            'CE_c': loss_CE_c.mean().item(),
            'CE_b': loss_CE_b.mean().item(),
            'GCE_b': loss_GCE_b.mean().item(),
        }

        z_b = self.encoder_bias(z)
        z_c = z - z_b.detach()
        pred = self.fc_final(z_c)
        loss_CE_c = self.CE(pred, y)
        loss_total += loss_CE_c.mean()

        if self.ampl_type == 'comb':
            loss_CE_pc, loss_CE_pb, loss_GCE_pb = self.combination(z_c, z_b, y)
        elif self.ampl_type == 'perm':
            loss_CE_pc, loss_CE_pb, loss_GCE_pb = self.permutation(z_c, z_b, y)
        else:
            raise ValueError(f'Invalid ampl_type: {self.ampl_type}')

        res.update({
            'CE_c': loss_CE_c.mean().item(),
            'CE_pc': loss_CE_pc.mean().item(),
            'CE_pb': loss_CE_pb.mean().item(),
            'GCE_pb': loss_GCE_pb.mean().item(),
        })

        if self.cf_coeff > 0:
            z_env = self.encoder_env(env)
        else:
            z_env = None

        losses_items = dict(loss_CE_c=loss_CE_c, loss_CE_b=loss_CE_b, loss_CE_pc=loss_CE_pc, loss_CE_pb=loss_CE_pb)
        amplitude = self.lambda_ampl(epoch, losses_items)

        if amplitude > 0:

            loss_cf = self.counterfactual(z_c, z_env, y)
            loss_total += loss_cf.mean()

            weight_amp = self.get_weight(loss_CE_pc, loss_CE_pb)
            loss_amp = weight_amp * loss_CE_pc + (1 - weight_amp) * loss_GCE_pb
            loss_total += amplitude * loss_amp.mean()

            res.update({
                'total': loss_total,
                'amplitude': amplitude,
                'amp': loss_amp.mean().item(),
                'weight_amp': weight_amp.mean().item(),
                'cf': loss_cf.mean().item(),
            })
        else:
            res.update({
                'total': loss_total,
                'amplitude': 0,
                'amp': 0,
                'weight_amp': 0,
                'cf': 0,
            })

        return pred, res

    def get_z(self, x):
        z = self.encoder(x)
        z_b = self.encoder_bias(z)
        z_c = z - z_b.detach()
        return z_c, z

    def forward(self, x):
        z = self.encoder(x)
        z_b = self.encoder_bias(z)
        z_c = z - z_b.detach()
        pred = self.fc_final(z_c)
        return pred


class GeneralizedCELoss(nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        assert 0 < q < 1
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)  # [N, C]
        # if np.isnan(p.mean().item()):
        #     raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # Yg: [N, 1], the probability on the channel of GT class
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach() ** self.q)  # * self.q
        # if np.isnan(Yg.mean().item()):
        #     raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss


class Amplification(nn.Module):
    def __init__(self, config_ampl, ampl_type):
        super().__init__()
        config_ampl = config_ampl.copy()
        _ = config_ampl.pop('num_epoch')
        assert len(config_ampl) == 2, f'Invalid config_ampl: {config_ampl}'
        self.kind = config_ampl['kind']
        self.pre_epoch = config_ampl['pre_epoch']
        if self.kind.startswith('Var'):
            assert ampl_type == 'comb', f'WRONG: amp={ampl_type}, kind={self.kind}, but should =comb'

    def forward(self, epoch, losses_items):
        if epoch < self.pre_epoch:
            return 0
        losses_items = {kk: vv.clone().detach() for kk, vv in losses_items.items()}
        if self.kind == 'VarDiv':
            loss_CE_pc = losses_items['loss_CE_pc']
            loss_CE_pb = losses_items['loss_CE_pb']
            assert len(loss_CE_pc.shape) == 2 and len(loss_CE_pb.shape) == 2
            var_c = torch.var(loss_CE_pc, dim=1)
            var_b = torch.var(loss_CE_pb, dim=1)
            weight = torch.mean(var_b / (var_c + var_b + 1e-8))
        elif self.kind == 'Var':  #
            loss_CE_pc = losses_items['loss_CE_pc']
            assert len(loss_CE_pc.shape) == 2
            weight = torch.mean(torch.sigmoid(torch.var(loss_CE_pc, dim=1)))
        else:
            raise ValueError(f'Invalid kind: {self.kind}')
        return weight.item()

