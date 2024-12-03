import pandas as pd
import numpy as np
import os
import shutil
import pickle
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, KFold

import networkx as nx
import torch
from torch_geometric.data import Data


save_dir = './../../dataset'
proc_dir = '/home/data2/MRIProcess'
split_list = list(range(2020, 2030))
npy_config_root = dict(
    sampler='ave',
    kernel=2,
)


def main():
    main_NACC()
    # main_NACCVal()
    # main_AIBL()
    # main_ADNI()
    # main_OASIS3()
    # main_ABIDE()
    # main_MouseMot()


def main_NACC():
    img = Images(data_base='NACC', npy_dir=os.path.join(proc_dir, 'NACC/nc_fast_final'))
    npy_config = dict(**npy_config_root, path=os.path.join(proc_dir, 'NACC/nc_fast_final'),
                      suffix='(1e-02;75;MinMax;clip).npy')
    img.save_to_pkl(npy_config)

    obj = PrepareData(data_base='NACC', data_name='DN0.3_image_3')
    Parallel(n_jobs=3)(delayed(obj.Split_TestValidTrain)(seed=seed, mode='patient') for seed in split_list)


def main_AIBL():
    img = Images(data_base='AIBL', npy_dir=os.path.join(proc_dir, 'AIBL/nc_fast_final'))
    npy_config = dict(**npy_config_root, path=os.path.join(proc_dir, 'AIBL/nc_fast_final'),
                      suffix='(1e-02;75;MinMax;clip).npy')
    img.save_to_pkl(npy_config)

    obj = PrepareData(data_base='AIBL', data_name='DN0.3_image_3')
    Parallel(n_jobs=3)(delayed(obj.Split_TestValidTrain)(seed=seed, mode='patient') for seed in split_list)


def main_ADNI():
    img = Images(data_base='ADNI', npy_dir=os.path.join(proc_dir, 'ADNI/nc_none_final'))
    npy_config = dict(**npy_config_root, path=os.path.join(proc_dir, 'ADNI/nc_none_final'),
                      suffix='(1e-02;75;MinMax;clip).npy')
    img.save_to_pkl(npy_config)

    obj = PrepareData(data_base='ADNI', data_name='DN0.3_image_3')
    Parallel(n_jobs=3)(delayed(obj.Split_TestValidTrain)(seed=seed, mode='patient') for seed in split_list)


def main_OASIS3():
    img = Images(data_base='OASIS3', npy_dir=os.path.join(proc_dir, 'OASIS3/nc_fast_final'))
    npy_config = dict(**npy_config_root, path=os.path.join(proc_dir, 'OASIS3/nc_fast_final'),
                      suffix='(1e-02;75;MinMax;clip).npy')
    img.save_to_pkl(npy_config)

    obj = PrepareData(data_base='OASIS3', data_name='DN0.3_image_3')
    Parallel(n_jobs=3)(delayed(obj.Split_TestValidTrain)(seed=seed, mode='patient') for seed in split_list)


def main_NACCVal():
    obj = PrepareData(data_base='NACC', data_name='DN0.3_value_2')
    Parallel(n_jobs=3)(delayed(obj.Split_TestValidTrain)(seed=seed, mode='patient') for seed in split_list)


def main_ABIDE():
    graph = Graphs(data_base='ABIDE', graph_dir=os.path.join(proc_dir, 'ABIDE/cpac_filt_noglobal'),
                   suffix='_cc200_graph1.npy')
    graph.save(mode='adj')

    obj = PrepareData(data_base='ABIDE', data_name='DN0.3_image_1')
    Parallel(n_jobs=3)(delayed(obj.Split_TestValidTrain)(seed=seed, mode='patient', id_col='SubID')
                       for seed in split_list)


def main_MouseMot():
    obj = PrepareData(data_base='MouseMot', data_name='DN0.3_v2')
    Parallel(n_jobs=3)(delayed(obj.Split_TestValidTrain)(seed=seed, mode='batch')
                       for seed in split_list)


class PrepareData:
    def __init__(self, data_base='NACC', data_name='DN0.3_value_1'):
        self.data_base = data_base
        self.data_name = data_name
        self.save_dir = os.path.join(save_dir, data_base, data_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.data_path = os.path.join(save_dir, data_base, data_name, 'Data.csv')
        src = os.getcwd().split('scripts')[0] + f'/handle_csv/{data_base}/'
        try:
            shutil.copy2(os.path.join(src, f'{data_name}_Data.csv'),
                         self.data_path)
            shutil.copy2(os.path.join(src, f'{data_name}_Stats.csv'),
                         self.data_path.replace('Data.csv', 'Stats.csv'))
        except FileNotFoundError:
            raise FileNotFoundError(f'{src} does not exist!')
        self.Data = pd.read_csv(self.data_path, index_col=0)
        print(self.Data['Label'].value_counts())

    def Split_TestValidTrain(self, seed, mode='patient', id_col='NACCID'):
        out_path = os.path.join(self.save_dir, f'{mode[0]}.split{seed}')
        # if os.path.isfile(os.path.join(out_path, 'cross5CV.pkl')):
        #     print(f'cross5CV.pkl exists in {out_path}')
        #     return
        os.makedirs(out_path, exist_ok=True)

        if mode == 'visit':
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            split = skf.split(X=self.Data['Label'], y=self.Data['Label'])
        elif mode == 'patient':
            # PatientID = [kk.split('.')[0] for kk in self.Data.index]
            if id_col in self.Data.columns:
                PatientID = self.Data[id_col]
            elif id_col == self.Data.index.name:
                PatientID = self.Data.index
            else:
                raise ValueError(f'Unknown id_col: {id_col}')
            PatientSet = list(np.unique(PatientID))
            groups = [PatientSet.index(s) for s in PatientID]
            skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            split = skf.split(X=self.Data['Label'], y=self.Data['Label'], groups=groups)
        elif mode == 'batch':
            BatchSet = list(np.unique(self.Data['batch']))
            groups = [BatchSet.index(s) for s in self.Data['batch']]
            skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            split = skf.split(X=self.Data['Label'], y=self.Data['Label'], groups=groups)
        else:
            raise ValueError(f'Unknown mode: {mode}')

        fold_idx = []
        for (_, test) in split:
            fold_idx.append(test)

        cross5CV = dict()
        for i in range(5):
            test = self.Data.index[fold_idx[i]].tolist()
            val = self.Data.index[fold_idx[(i + 1) % 5]].tolist()
            train = []
            for kk in range(5):
                if kk in [i, (i + 1) % 5]:
                    continue
                train.extend(self.Data.index[fold_idx[kk]])
            for dd in ['train', 'val', 'test']:
                counts = self.Data.loc[eval(dd), 'Label'].value_counts()
                counts_str = ', '.join([f'{kk}:{counts[kk]}' for kk in counts.keys()])
                print(seed, f'fold{i}', dd, '-' * 5 + '\t', counts_str)
                cross5CV[f'{dd}{i}'] = eval(dd)

        if os.path.isfile(os.path.join(out_path, 'cross5CV.pkl')):
            with open(os.path.join(out_path, 'cross5CV.pkl'), 'rb') as f:
                old = pickle.load(f)
            assert old == cross5CV
            print(f'cross5CV.pkl exists in {out_path} and is equal')
        else:
            with open(os.path.join(out_path, 'cross5CV.pkl'), 'wb') as f:
                pickle.dump(cross5CV, f)


class Transform:
    def __init__(self, config):
        config = config.copy()
        self.sampler, self.kernel = config.pop('sampler'), config.pop('kernel')
        assert isinstance(self.kernel, int) and self.kernel >= 1
        if self.kernel == 1:
            assert self.sampler == 'ave'
        # self.contrast, self.bright, self.noise = config.pop('contrast'), config.pop('bright'), config.pop('noise')
        # assert 0 <= self.contrast < 1 and 0 <= self.bright < 1 and 0 <= self.noise < 1
        assert len(config) == 0
        self.dims = [xx // self.kernel for xx in [182, 218, 182]]

    def down_sampling(self, image0):
        if self.kernel == 1:
            return image0
        if self.sampler == 'ave':
            app = lambda x: np.mean(x.flatten())
        elif self.sampler == 'max':
            app = lambda x: np.max(x.flatten())
        elif self.sampler == 'min':
            app = lambda x: np.min(x.flatten())
        elif self.sampler == 'random':
            app = lambda x: x.flatten()[np.random.randint(0, self.kernel ** 3)]
        else:
            raise ValueError(f'Unknown sampler: {self.sampler}')

        X, Y, Z = image0.shape
        image = np.zeros(self.dims).flatten()
        corners = np.array([[x, y, z]
                            for x in range(0, X - self.kernel, self.kernel)
                            for y in range(0, Y - self.kernel, self.kernel)
                            for z in range(0, Z - self.kernel, self.kernel)])
        for idx, (i, j, k) in enumerate(corners):
            image[idx] = app(image0[i: i + self.kernel, j: j + self.kernel, k: k + self.kernel])
        image = image.reshape(self.dims)

        # for i in range(image.shape[0]):
        #     for j in range(image.shape[1]):
        #         for k in range(image.shape[2]):
        #             ii, jj, kk = [[xx * self.kernel, (xx + 1) * self.kernel] for xx in [i, j, k]]
        #             if self.sampler == 'ave':
        #                 image[i, j, k] = np.mean(image0[ii[0]: ii[1], jj[0]: jj[1], kk[0]: kk[1]].flatten())
        #             elif self.sampler == 'max':
        #                 image[i, j, k] = np.max(image0[ii[0]: ii[1], jj[0]: jj[1], kk[0]: kk[1]].flatten())
        #             elif self.sampler == 'min':
        #                 image[i, j, k] = np.min(image0[ii[0]: ii[1], jj[0]: jj[1], kk[0]: kk[1]].flatten())
        #             elif self.sampler == 'random':
        #                 idx = np.random.randint(0, self.kernel ** 3)
        #                 image[i, j, k] = image0[ii[0]: ii[1], jj[0]: jj[1], kk[0]: kk[1]].flatten()[idx]
        return image

    # def change_contrast(self, image0):
    #     return image0 * (1 + np.random.uniform(-self.contrast, self.contrast))  # [1-c, 1+c]
    #
    # def change_brightness(self, image0):
    #     return image0 + np.random.uniform(-self.bright, self.bright)
    #
    # def add_noise(self, image0):
    #     sig = np.random.uniform(0, self.noise)
    #     return image0 + np.random.normal(0, sig, image0.shape)

    def apply(self, image):
        image = self.down_sampling(image)
        # if self.contrast > 0:
        #     image = self.change_contrast(image)
        # if self.bright > 0:
        #     image = self.change_brightness(image)
        # if self.noise > 0:
        #     image = self.add_noise(image)
        return image


class Images:
    def __init__(self, data_base, npy_dir):
        self.data_base = data_base
        self.save_dir = os.path.join(save_dir, data_base)
        self.npy_list = set([kk.split('(')[0] for kk in os.listdir(npy_dir) if kk.endswith('.npy')])
        self.sub_list = self.get_subjects()
        os.makedirs(self.save_dir, exist_ok=True)
    
    def get_subjects(self):
        data = pd.read_csv(os.path.join(os.getcwd().split('scripts')[0], 'handle_csv', self.data_base, 'mri_path_add.csv'))
        npy_list = set(self.npy_list) & set([kk.split('.nii')[0] for kk in data['link']])
        data['add'] = data['link'].apply(lambda x: x.split('.nii')[0])
        data = data[data['add'].isin(npy_list)].reset_index(drop=True)
        print('Sample number (have MRI files): ', len(data))  # 6520
        # # 添加被试编号
        # subjects = []
        # for i in range(len(data)):
        #     SubID = data.loc[i, 'link'].split('.nii')[0]
        #     assert '{d[NACCID]}.{d[NACCVNUM]}.{d[NACCMNUM]}'.format(d=data.loc[i, :]) == SubID.rsplit('.', 1)[0]
        #     subjects.append(SubID)
        return npy_list
    
    def save_to_pkl(self, npy_config):
        npy_config = npy_config.copy()
        npy_path = npy_config.pop('path')
        suffix = npy_config.pop('suffix')
        trans = Transform(npy_config)
        data_all = dict()

        # for sub in tqdm(self.Data.index[:5]):
        #     time1 = time.time()
        #     npy = np.load(os.path.join(npy_path, sub + suffix), mmap_mode='r').astype(np.float32)
        #     print(f'Load {sub} in {time.time() - time1:.2f}s')
        #     time1 = time.time()
        #     npy = trans.apply(npy)
        #     print(f'finish transform in {time.time() - time1:.2f}s')
        #     data_all[sub] = npy

        def OneSample(subject):
            npy = np.load(os.path.join(npy_path, subject + suffix), mmap_mode='r').astype(np.float32)
            npy = trans.apply(npy)
            return subject, npy

        packages = Parallel(n_jobs=20)(delayed(OneSample)(sub) for sub in tqdm(self.sub_list))
        data_all.update({sub: npy for sub, npy in packages})

        # save_name = (self.data_name + f'_({suffix})_' +
        #              '{n[sampler]}{n[kernel]:d}_{n[contrast]:g}_{n[bright]:g}_{n[noise]:g}'.format(n=npy_config))
        save_name = f"{suffix}_{npy_config['sampler']}_{npy_config['kernel']:d}"
        with open(os.path.join(self.save_dir, save_name + '.pkl'), 'wb') as f:
            pickle.dump(data_all, f)
        # np.savez(os.path.join(self.save_dir, save_name + '.npz'), **data_all)


class Graphs:
    def __init__(self, data_base, graph_dir, suffix):
        self.data_base = data_base
        self.save_dir = os.path.join(save_dir, data_base)
        self.graph_dir = graph_dir
        self.suffix = suffix
        self.graph_list = set([kk.split('_')[0] for kk in os.listdir(self.graph_dir) if kk.endswith(suffix)])

        self.sub_list = self.get_subjects()
        os.makedirs(self.save_dir, exist_ok=True)

    def get_subjects(self):
        data = pd.read_csv(os.path.join(os.getcwd().split('scripts')[0], 'handle_csv', self.data_base, 'Table_graph.csv'))
        sub_list = set(self.graph_list) & set([str(kk) for kk in data['SUB_ID']])  # 存在图数据 且 存在表格数据
        return sorted(list(sub_list))

    def save(self, mode):
        save_name = f"{os.path.basename(self.graph_dir.strip('/'))}{self.suffix}_{mode}"

        def OneSample(subject):
            data0 = np.load(os.path.join(self.graph_dir, subject + self.suffix))
            num_node = data0.shape[0]
            adj = data0[:, :num_node]
            node = data0[:, num_node:]
            if mode == 'adj':
                return subject, node, adj
            elif mode == 'nx':
                return subject, node, matrix2edges(adj)

        packages = Parallel(n_jobs=20)(delayed(OneSample)(sub) for sub in tqdm(self.sub_list))
        if mode == 'adj':
            data_all = {sub: [node, adj] for sub, node, adj in packages}
            with open(os.path.join(self.save_dir, save_name + '.pkl'), 'wb') as f:
                pickle.dump(data_all, f)

        elif mode == 'nx':
            sub_list, att_list, edge_att_list, edge_index_list, batch, pseudo = [], [], [], [], [], []
            for j in range(len(packages)):  # packages[j][0-2]分别是：边权重、边索引、节点特征
                sub, att, (edge_att, edge_index) = packages[j]
                num_nodes = att.shape[0]
                sub_list.append(sub)
                att_list.append(att)  # 节点特征
                edge_att_list.append(edge_att)  # 边权重
                edge_index_list.append(edge_index + j * num_nodes)  # 边索引
                batch.append([j] * num_nodes)  # batch的序号复制，次数=节点个数
                pseudo.append(np.diag(np.ones(num_nodes)))  # 就是文中的r_i，一个单位阵，用于表示ROI的编号

            att_arr = np.concatenate(att_list, axis=0)  # (400, 200) 沿dim0拼接
            edge_att_arr = np.concatenate(edge_att_list)  # 79600 样本个数×39800
            edge_index_arr = np.concatenate(edge_index_list, axis=1)  # (2, 79600)
            pseudo_arr = np.concatenate(pseudo, axis=0)  # (400, 200)

            att_torch = torch.from_numpy(att_arr).float()
            edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
            edge_index_torch = torch.from_numpy(edge_index_arr).long()
            pseudo_torch = torch.from_numpy(pseudo_arr).float()
            batch_torch = torch.from_numpy(np.hstack(batch)).long()  # 400 200个0和200个1
            data = Data(x=att_torch, edge_attr=edge_att_torch, edge_index=edge_index_torch, pos=pseudo_torch,
                        sub=sub_list)

            data, slices = split_graph(data, batch_torch)  # 上面line26，用于把样本分开
            torch.save((data, slices), os.path.join(self.save_dir, save_name + '.pt'))


def matrix2edges(matrix):
    G = nx.convert_matrix.from_numpy_array(matrix)  # G[j,i]=pcorr[i,j]注意行列是反的
    A = nx.to_scipy_sparse_array(G)  # csr_matrix稀疏矩阵 (row,col)坐标对应一个边权重
    adj = A.tocoo()  # coo_matrix
    edge_att = np.zeros(len(adj.row))  # 一共4000条边，记录每一个边权重
    for i in range(len(adj.row)):
        edge_att[i] = matrix[adj.row[i], adj.col[i]]
    edge_index = np.stack([adj.row, adj.col])  # (2, 40000) 边的行列序号
    return edge_att, edge_index


def split_graph(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)  # bincount得到batch编号的计数，即200和200，cumsum是累加，得到[200,400]
    node_slice = torch.cat([torch.tensor([0]), node_slice])  # 最前面加一个0，得到[0,200,400]

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])  # [0,39800,79600]

    # Edge indices should start at zero for every graph.  输入的边序号是总序号，减去累加的节点数，使每个图的边索引都从0开始
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)  # 比样本数多1
    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices


if __name__ == '__main__':
    main()
