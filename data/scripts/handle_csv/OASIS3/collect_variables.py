import pandas as pd
import numpy as np
import os
import shutil
import time
from itertools import product
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
import pickle
from collect_samples import Logger, print_format, isNaN


class CollectVariables:
    def __init__(self):
        self.mri = 'image_3'
        self.missing_threshold = 0.3
        self.ref_csv = '/handle_csv/NACC/DN0.3_image_3_Stats.csv'
        self.Variables = pd.read_csv(os.getcwd().split('scripts')[0] + self.ref_csv)
        self.npy_list = [kk.split('.nii')[0] for kk in os.listdir('MRIProcess/OASIS3/nc_fast_final')]

        self.Save_path = os.getcwd().split('scripts')[0] + '/handle_csv/OASIS3/'  # handled csvs will be saved here
        self.save_log = True

        self.backup()

    def backup(self):
        timestamp = time.strftime('%Y%m%d%H%M%S')
        shutil.copy2('collect_variables.py', os.path.join(os.getcwd(), 'backup', f"collect_variables_{timestamp}.py"))
        self.Variables.to_csv(os.path.join(os.getcwd(), 'backup', f'Variables_{timestamp}.csv'), index=False)

        if self.save_log:
            sys.stdout = Logger("convert_missing", timestamp)

        print('\n' + '#' * 10 + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' Collect Variables ' + '#' * 10)
        print(f"Name: {self.mri}\nMissing threshold: {self.missing_threshold}\n"
              f"Reference csv: {self.ref_csv}\nVariables shape: {self.Variables.shape}\n")

    def main(self):
        dataAdd = pd.read_csv(os.path.join(self.Save_path, f'Merged_Data.csv'))
        # dataAdd = dataAdd.dropna(subset=['Label'], axis=0).reset_index(drop=True)
        # dataAdd = dataAdd[dataAdd['Use'] == 1].reset_index(drop=True)
        
        dataAdd['NACCID'] = [kk.split('OAS3')[1] for kk in dataAdd['OASISID']]
        dataAdd['NACCADC'] = 1

        npy_list = set(self.npy_list) & set([kk.split('.nii')[0] for kk in dataAdd['link']])
        dataAdd = dataAdd[dataAdd['SubID'].isin(npy_list)].reset_index(drop=True)
        print('Sample number (have MRI files): ', len(dataAdd), '\n')

        data = pd.DataFrame(columns=['SubID'] + self.Variables['Variable name'].tolist())
        for col in data.columns:
            if col in dataAdd.columns:
                data[col] = dataAdd[col]
            else:
                print(f'Column {col} NOT exist in dataAdd.columns.')
                data[col] = np.nan

        # self.drop_missing_samples(data.copy())
        self.drop_missing(data.copy())

    def drop_missing_samples(self, data):
        print(f"\nData shape: {data.shape}")
        data.index = data['SubID']
        data.drop(columns=['SubID'], inplace=True)

        missS = pd.isna(data).sum(axis=1).to_dict()
        S = int(data.shape[0] * self.missing_threshold)
        missS = [kk for kk, vv in missS.items() if vv > S]
        print(' ' * 9 + '\tOrigin\tDrop\tRemain')
        print(f'Samples  \t{len(data)}\t{len(missS)}\t{len(data) - len(missS)}')
        print('Final missing samples: ', missS)
        data_dropNaN = data.drop(missS, axis=0)
        print(f'Data shape after drop missing ({self.missing_threshold}): {data_dropNaN.shape}')
        assert data_dropNaN.shape[0] == len(data) - len(missS)
        data_dropNaN.to_csv(os.path.join(self.Save_path, f'DN{self.missing_threshold}_{self.mri}S_Data.csv'), index=True)

        assert set(data_dropNaN.columns) == set(self.Variables['Variable name'])
        stats_scope = self.Variables.copy()
        for i in stats_scope.index:
            col, col_type = stats_scope.loc[i, ['Variable name', 'Type']]
            if col == 'NACCAGE':
                col_type = 'continuous (float)'
                stats_scope.iloc[i]['Type'] = col_type
            values = data_dropNaN[col].values
            # print(col, col_type, values[:5])
            col_range, Min, Max = check_variables(values, col_type)
            stats_scope.loc[i, ['Range', 'Min', 'Max']] = np.array([col_range, Min, Max], dtype=object)

        stats_scope.to_csv(os.path.join(self.Save_path, f'DN{self.missing_threshold}_{self.mri}S_Stats.csv'),
                           index=False)

    def drop_missing(self, data):
        print(f"\nData shape: {data.shape}")
        data.index = data['SubID']
        data.drop(columns=['SubID'], inplace=True)
        data.dropna(axis=1, how='all', inplace=True)
        print('After dropAll NaN columns, data.shape: ', data.shape)

        missF, missS = count_missing_ratio(data)
        F, S = [int(kk * self.missing_threshold) for kk in data.shape]
        missF = {kk: vv for kk, vv in missF.items() if vv > F}
        missS = {kk: vv for kk, vv in missS.items() if vv > S}

        print('\n' + '-' * 10 + ' Summary ' + '-' * 10)
        print(' ' * 9 + '\tOrigin\tDrop\tRemain')
        print(f'Variables\t{data.shape[1]}\t{len(missF)}\t{data.shape[1] - len(missF)}')
        print(f'Samples  \t{data.shape[0]}\t{len(missS)}\t{data.shape[0] - len(missS)}')
        print('Final missing features: ', missF)
        print('Final missing samples: ', missS)
        data_dropNaN = data.drop(missS, axis=0).drop(missF, axis=1)
        print(f'Data shape after drop missing ({self.missing_threshold}): {data_dropNaN.shape}')
        assert data_dropNaN.shape[0] == len(data) - len(missS) and data_dropNaN.shape[1] == data.shape[1] - len(missF)
        data_dropNaN.to_csv(os.path.join(self.Save_path, f'DN{self.missing_threshold}_{self.mri}_Data.csv'), index=True)

        # assert set(data_dropNaN.columns) == set(self.Variables['Variable name'])
        del_index = []
        stats_scope = self.Variables.copy()
        for i in stats_scope.index:
            col, col_type = stats_scope.loc[i, ['Variable name', 'Type']]
            if col not in data_dropNaN.columns:
                del_index.append(i)
                continue
            if col == 'NACCAGE':
                col_type = 'continuous (float)'
                stats_scope.iloc[i]['Type'] = col_type
            values = data_dropNaN[col].values
            col_range, Min, Max = check_variables(values, col_type)
            stats_scope.loc[i, ['Range', 'Min', 'Max']] = np.array([col_range, Min, Max], dtype=object)
        stats_scope.drop(index=del_index, inplace=True)

        stats_scope.to_csv(os.path.join(self.Save_path, f'DN{self.missing_threshold}_{self.mri}_Stats.csv'),
                           index=False)


# ############# functions #############


def count_missing_ratio(data):
    missF = pd.isna(data).sum(axis=0)
    missS = pd.isna(data).sum(axis=1)
    return missF.to_dict(), missS.to_dict()


def check_variables(values, col_type):
    col_values, is_float = convert_float(values)
    if not isinstance(col_values[0], str) and np.isnan(col_values).all():
        col_range = 'nan~nan'
        Min, Max = np.nan, np.nan

    elif 'int' in col_type:
        assert is_float
        Min, Max = np.nanmin(col_values), np.nanmax(col_values)
        col_set = set([kk for kk in col_values if not isNaN(kk)])
        col_values, is_categorical = convert_categorical(col_set)
        assert is_categorical
        if col_type == 'category (int)':
            col_range = '[' + ', '.join([f'{int(kk):d}' for kk in sorted(col_set)]) + ']'
        elif col_type == 'continuous (int)':
            if isNaN(Min) or isNaN(Max):
                col_range = 'nan~nan'
            else:
                col_range = f'{int(Min):d}~{int(Max):d}'
        else:
            raise ValueError(f'Unknown col_type: {col_type}')

    elif 'float' in col_type:
        assert is_float
        Min, Max = np.nanmin(col_values), np.nanmax(col_values)
        col_range = f'{Min:.3f}~{Max:.3f}'

    else:
        assert not is_float
        Min, Max = -999, -999
        col_set = set([kk for kk in col_values if kk != ''])
        if col_type == 'category (str)':
            col_range = '[' + ', '.join(sorted(col_set)) + ']'
        elif col_type == 'None (str)':
            col_range = 'SKIP'
        else:
            raise ValueError(f'Unknown col_type: {col_type}')

    return [col_range, Min, Max]


def convert_float(s):
    try:
        convert = [float(ss) for ss in s]
        return convert, True
    except ValueError:
        convert = [str(ss) for ss in s]
        return convert, False


def convert_categorical(s):
    s, flag = convert_float(s)
    if not flag:
        return s, False
    convert = [MyInt(ss) for ss in s]
    if None in convert:
        return s, False
    else:
        return convert, True


def MyInt(s):
    close = int(np.round(s))
    if abs(s - close) < 1e-6:
        return close
    else:
        return None


if __name__ == '__main__':
    obj = CollectVariables()
    obj.main()
