# Copyright (c) 2019 Mwiza Kunda
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, , Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Reference: https://github.com/xxlya/BrainGNN_Pytorch
"""

from nilearn import datasets
import argparse
import os
import glob
import shutil
import re
import numpy as np
import random
import scipy.io as sio
import pickle
from nilearn import connectome
from scipy import sparse
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='Download ABIDE data, compute connectivity, and construct graph.')
    parser.add_argument('--data', default='/home/data2/PublicData/ABIDE/', type=str,
                        help='Directory containing the ABIDE data.')
    parser.add_argument('--save', default='./../raw_pre/', type=str,
                        help='Directory to save the processed data.')
    parser.add_argument('--sub_path', default='./subjects_IDs.txt', type=str,
                        help='Path to the file containing the subject IDs.')

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--pipeline', default='cpac', type=str,
                        help='Pipeline to preprocess ABIDE data. Available options are ccs, cpac, dparsf and niak.'
                             ' default: cpac.')
    parser.add_argument('--atlas', default='cc200', type=str,
                        help='Brain parcellation atlas. Options: aal, ho, cc200,'
                             ' default: cc200.')
    parser.add_argument('--filt', default=True, type=str2bool,
                        help='band_pass_filtering, default: True.')
    parser.add_argument('--reg', default=False, type=str2bool,
                        help='global_signal_regression, default: False.')
    parser.add_argument('--quality', default=True, type=str2bool,
                        help='quality_checked, default: True.')
    # parser.add_argument('--mode', default='graph1', type=str,
    #                     help='Graph construction mode. Options: graph1, graph2, etc. Default: graph1')

    args = parser.parse_args()
    print(args)

    setting_seed(args.seed)
    prefix = ('filt' if args.filt else 'nofilt') + '_' + ('global' if args.reg else 'noglobal')
    data_dir = os.path.join(args.data, 'ABIDE_pcp', args.pipeline, prefix)
    save_dir = os.path.join(os.getcwd(), args.save, args.pipeline, prefix)
    graph_dir = os.path.join(os.getcwd(), args.save, '..', f'{args.pipeline}_{prefix}')
    files = ['rois_' + args.atlas]

    # phenotype_path = os.path.join(args.data, 'ABIDE_pcp', 'Phenotypic_V1_0b_preprocessed1.csv')
    if os.path.isfile(args.sub_path):
        subject_IDs = np.genfromtxt(args.sub_path, dtype=str).tolist()
    else:
        subject_IDs = [str(int(re.findall(r'_(005\d+)_', s)[0])) for s in os.listdir(data_dir)]
    print(subject_IDs[:5])

    # download(args, files)

    reader = Reader(data_path=data_dir, subject_IDs=subject_IDs, save_path=save_dir)
    calc_connect(args, reader, files)

    batch_graphs(args, reader, graph_dir)


def download(args, files):
    # Download database files
    abide = datasets.fetch_abide_pcp(
                data_dir=args.data, pipeline=args.pipeline, derivatives=files,
                band_pass_filtering=args.filt, global_signal_regression=args.reg, quality_checked=args.quality,
            )
    # 1D original file, name example：Caltech_0051461_rois_cc200.1D
    # quality_checked=True -- 871
    # quality_checked=False -- 1035
    # https://blog.csdn.net/qq_30049011/article/details/113182835


def calc_connect(args, reader, files):
    file_mapping = {kk: f'{kk}.1D' if kk.startswith('rois') else f'{kk}.nii.gz' for kk in files}
    # Create a folder for each subject
    for s, fname in zip(reader.subject_IDs, reader.fetch_filenames(files[0])):
        subject_folder = os.path.join(reader.save_path, s)
        os.makedirs(subject_folder, exist_ok=True)

        # Get the base filename for each subject
        base = fname.split(files[0])[0]

        # Move each subject file to the subject folder
        for fl in files:
            if not os.path.exists(os.path.join(subject_folder, base + file_mapping[fl])):
                shutil.copy2(base + file_mapping[fl], subject_folder)

    time_series = reader.get_timeseries(args.atlas)

    # Compute and save connectivity matrices
    reader.subject_connectivity(time_series, args.atlas, 'correlation')
    reader.subject_connectivity(time_series, args.atlas, 'partial correlation')
    # save file name example：51461_cc200_correlation.mat


def batch_graphs(args, reader, graph_dir):
    reader.batch_graphs(graph_dir, args.atlas)


def single_graph(data_dir, save_dir, atlas_name, subject):
    if os.path.isfile(os.path.join(save_dir, f'{subject}_{atlas_name}.pkl')):
        return

    os.makedirs(save_dir, exist_ok=True)
    # 'corr': (200, 200), 'pcorr': (200, 200)
    # read edge and edge attribute
    corr = os.path.join(data_dir, subject, f"{subject}_{atlas_name}_correlation.mat")
    att = safe_arctanh(sio.loadmat(corr)['connectivity'])
    att[np.diag_indices_from(att)] = 0

    pcorr = os.path.join(data_dir, subject, f"{subject}_{atlas_name}_partial_correlation.mat")
    pcorr = np.abs(safe_arctanh(sio.loadmat(pcorr)['connectivity']))
    pcorr[np.diag_indices_from(pcorr)] = 0

    stack = np.concatenate([pcorr, att], axis=1)
    np.save(os.path.join(save_dir, f'{subject}_{atlas_name}_graph1.npy'), stack)
    # with open(os.path.join(save_dir, f'{subject}_{atlas_name}_graph1.pkl'), 'wb') as f:
    #     pickle.dump({'adj': pcorr, 'node': att}, f)


def safe_arctanh(x, eps=1e-7):
    x = np.clip(x, -1 + eps, 1 - eps)
    return np.arctanh(x)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setting_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


class Reader:
    def __init__(self, data_path, save_path, subject_IDs=None):
        """
        :param data_path:
        :param save_path: specify path to save the matrix
        :param subject_IDs: list of short subject IDs in string format
        """
        self.data_path = data_path
        self.save_path = save_path
        self.subject_IDs = subject_IDs

    def fetch_filenames(self, file_type):
        """
        :param file_type: must be one of the available file types
        :returns:
            filenames: list of filetypes (same length as subject_list)
        """

        filemapping = f'_{file_type}.1D' if file_type.startswith('rois') else f'_{file_type}.nii.gz'
        # The list to be filled
        filenames = []

        # Fill list with requested file paths
        for i in range(len(self.subject_IDs)):
            os.chdir(self.data_path)
            try:
                try:
                    os.chdir(self.data_path)
                    filenames.append(glob.glob('*' + self.subject_IDs[i] + filemapping)[0])
                except:
                    os.chdir(self.data_path + '/' + self.subject_IDs[i])
                    filenames.append(glob.glob('*' + self.subject_IDs[i] + filemapping)[0])
            except IndexError:
                filenames.append('N/A')
        return filenames

    # Get timeseries arrays for list of subjects
    def get_timeseries(self, atlas_name, silence=True):
        """
        :param atlas_name: name of the atlas used for parcellation
        :param silence:     print the file being read
        :returns:
            time_series:    list of timeseries arrays, each of shape (timepoints x regions)
        """

        timeseries = []
        for i in range(len(self.subject_IDs)):
            subject_folder = os.path.join(self.save_path, self.subject_IDs[i])
            ro_file = [f for f in os.listdir(subject_folder) if f.endswith(f'_rois_{atlas_name}.1D')]
            fl = os.path.join(subject_folder, ro_file[0])
            if not silence:
                print("Reading timeseries file %s" % fl)
            timeseries.append(np.loadtxt(fl, skiprows=0))

        return timeseries

    #  compute connectivity matrices
    def subject_connectivity(self, timeseries, atlas_name, kind, save=True):
        """
        :param timeseries:  timeseries table for subject (timepoints x regions)
        :param atlas_name: name of the atlas used for parcellation
        :param kind:        the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        :param save:        save the connectivity matrix to a file
        :returns
            connectivity:   connectivity matrix (regions x regions)
        """

        if kind in ['correlation', 'partial correlation']:
            conn_measure = connectome.ConnectivityMeasure(kind=kind)
            connectivity = conn_measure.fit_transform(timeseries)
        else:
            raise ValueError(f'Connectivity method: {kind} not supported')

        if save:
            if kind in ['correlation', 'partial correlation']:
                kind0 = '_'.join(kind.split())
                for i, subj_id in enumerate(self.subject_IDs):
                    subject_file = os.path.join(self.save_path, subj_id, f"{subj_id}_{atlas_name}_{kind0}.mat")
                    sio.savemat(subject_file, {'connectivity': connectivity[i]})
        return connectivity

    def batch_graphs(self, graph_path, atlas_name):
        os.makedirs(graph_path, exist_ok=True)
        # parallel computing
        Parallel(n_jobs=4)(delayed(single_graph)(self.save_path, graph_path, atlas_name, sub)
                           for sub in self.subject_IDs)


if __name__ == '__main__':
    main()
