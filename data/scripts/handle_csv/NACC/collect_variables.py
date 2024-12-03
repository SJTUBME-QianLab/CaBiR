import pandas as pd
import numpy as np
import os
import shutil
import time
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
import pickle
from collect_samples import Logger, print_format, isNaN


class CollectVariables:
    def __init__(self):
        head = {
            'value': ['Label', 'SubID', 'NACCID'],
            'image': ['Label', 'SubID', 'NACCID'],
        }
        InfoAdd = {
            'value_2': ['Label', 'NACCADC', 'MRIFIELD', 'MRIMANU', 'MRIMODL'],
            'image_3': ['Label', 'NACCADC', 'MRIFIELDj', 'MRIMANUj', 'MRIMODLj', 'MRITHICKj'],
        }
        USE = {
            'value_2': ['Demo', 'Func', 'MedH', 'Psychia', 'Psycho', 'MRIValue'],
            'image_3': ['Demo', 'Func', 'MedH', 'Psychia', 'Psycho'],
        }
        self.mri = 'value_2'
        # self.mri = 'image_3'
        self.missing_threshold = 0.3

        self.Variables = pd.read_excel('PublicData/NACC/NACC_DataWash.xlsx', sheet_name='Use1')
        self.RawTable_path = 'PublicData/NACC/CSV/'  # csv files downloaded from NACC
        if self.mri.startswith('image'):
            self.npy_list = [kk.split('(mean;8).npy')[0] for kk in os.listdir('MRIProcess/NACC/nc_fast_final')]

        self.Save_path = os.getcwd().split('scripts')[0] + '/handle_csv/NACC/'  # handled csvs will be saved here
        self.save_log = True
        self.combine_mode = 'replace'  # [replace, modify, add, drop]
        self.head = head[self.mri.split('_')[0]]
        self.InfoAdd = InfoAdd[self.mri]
        self.USE = USE[self.mri]

        self.backup()

    def backup(self):
        timestamp = time.strftime('%Y%m%d%H%M%S')
        os.makedirs(os.path.join(os.getcwd(), 'backup'), exist_ok=True)
        for file in os.listdir(os.getcwd()):
            if os.path.isdir(file):
                continue
            elif file.endswith('.log'):
                shutil.copy2(file, os.path.join(os.getcwd(), 'backup', file))
            else:
                name, suffix = os.path.splitext(file)
                new_name = f"{name}_{timestamp}{suffix}"
                shutil.copy2(file, os.path.join(os.getcwd(), 'backup', new_name))
        shutil.copy2('collect_variables.py', os.path.join(os.getcwd(), 'backup', f"collect_variables_{timestamp}.py"))
        self.Variables.to_csv(os.path.join(os.getcwd(), 'backup', f'Variables_{timestamp}.csv'), index=False)

        if self.save_log:
            sys.stdout = Logger("convert_missing", timestamp)

        print('\n' + '#' * 10 + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' Collect Variables ' + '#' * 10)
        print(f"Backup timestamp: {timestamp}\nName: {self.mri}\nMissing threshold: {self.missing_threshold}\n"
              f"Combine mode: {self.combine_mode}\nhead: {self.head}\nInfoAdd: {self.InfoAdd}\nUSE: {self.USE}")

    def main(self):
        if self.mri.startswith('image'):
            data = self.load_withImages()
        elif self.mri.startswith('value'):
            data = self.load_withValues()
        else:
            raise ValueError(f'Unknown self.mri: {self.mri} (must be image OR value)')

        # merge tables
        RawTable = pd.read_csv(os.path.join(self.RawTable_path, 'investigator_nacc63.csv'))
        mriTable = pd.read_csv(os.path.join(self.RawTable_path, 'investigator_mri_nacc63.csv'))
        del_MRI_col = ['FRONTGRY', 'FRONTWHT', 'FRONTCSF', 'OCCIPGRY', 'OCCIPWHT', 'OCCIPCSF',
                       'PARGRY', 'PARWHT', 'PARCSF', 'TEMPGRY', 'TEMPWHT', 'TEMPCSF']
        mriTable.drop(del_MRI_col, axis=1, inplace=True)
        dataAdd = pd.merge(data, RawTable, how='left')
        dataAdd = pd.merge(dataAdd, mriTable, how='left')

        # select specific variables
        self.Variables = self.modal_feature()
        dataAdd = dataAdd[self.head + self.Variables['Variable name'].tolist()]
        try:
            assert '_x' not in dataAdd.columns and '_y' not in dataAdd.columns
            pd.testing.assert_frame_equal(dataAdd[['SubID']], data[['SubID']])
            del data, RawTable
        except AssertionError:
            print([kk for kk in dataAdd.columns if '_x' in kk])
            print('Merge error! Please check the columns that have the same name yet different values.')
            raise

        # missing value code convert to NaN
        self.convert_missing(dataAdd)

    def load_withImages(self):
        data = pd.read_csv(os.path.join(self.Save_path, 'mri_path_add.csv'))
        npy_list = set([kk + '.nii' for kk in self.npy_list]) & set(data['link'].tolist())
        data = data[data['link'].isin(npy_list)].reset_index(drop=True)
        print('Sample number (have MRI files): ', len(data))  # 6520

        data.insert(loc=list(data.columns).index('NACCID') + 1, column='SubID', value='')
        for i in range(len(data)):
            SubID = data.loc[i, 'link'].split('.nii')[0]
            assert '{d[NACCID]}.{d[NACCVNUM]}.{d[NACCMNUM]}'.format(d=data.loc[i, :]) == SubID.rsplit('.', 1)[0]
            data.loc[i, 'SubID'] = SubID
        assert len(set(data['SubID'])) == len(data)
        return data

    def load_withValues(self):
        data = pd.read_csv(os.path.join(self.Save_path, 'Table_MRI.csv'))
        data = data.query('NACCMVOL==1').reset_index(drop=True)
        print('Sample number (have MRI values): ', len(data))  # 3443

        data.insert(loc=list(data.columns).index('NACCID') + 1, column='SubID', value='')
        for i in range(len(data)):
            SubID = data.loc[i, 'NACCID'] + (
                '.{:d}.{:d}'.format(*[int(data.loc[i, kk]) for kk in ['NACCVNUM', 'NACCMNUM']]))
            data.loc[i, 'SubID'] = SubID
        assert len(set(data['SubID'])) == len(data)
        return data

    def modal_feature(self):
        Variables = self.Variables.dropna(subset=['Group'], axis=0).reset_index(drop=True)
        # ALL: Demo, Diagnosis, Func, Info, MedH, MRIInfo, MRIValue, Psychia, Psycho, Score
        # NOT USE: Diagnosis, Info, MRIInfo, Score (too many NaN)
        if self.mri == 'image_3' or self.mri == 'value_2':
            assert 'CDRSUM' not in Variables and 'CDRGLOB' not in Variables

        modal_feature = pd.DataFrame()
        for col in self.InfoAdd:
            info = Variables[Variables['Variable name'] == col].copy()
            info.loc[:, 'Group'] = 'Info'
            modal_feature = pd.concat([modal_feature, info], axis=0)
        for modal in self.USE:
            modal_feature = pd.concat([modal_feature, Variables[Variables['Group'] == modal]], axis=0)
        return modal_feature

    def convert_missing1(self, data):
        print(f"\nData shape: {data.shape}")

        MissingTags = pd.Series(self.Variables['MissingTag'].values, index=self.Variables['Variable name'])
        print('\n' + '-' * 10 + ' Missing condition for every group ' + '-' * 10)
        data.index = data['SubID']
        DataAll = pd.DataFrame(index=data.index)
        Note = dict()
        minus = 0
        for g, c in self.Variables['Group'].value_counts().items():
            print(f'Group: {g}, Variable number: {c}')

            cols = self.Variables[self.Variables['Group'] == g]['Variable name'].tolist()
            dataset = data.loc[:, cols]
            for cc in cols:
                dataset[cc] = replace_missing(MissingTags[cc], dataset[cc])

            if g == 'MedH' and self.combine_mode is not None:
                dataset, num = self.combine_MedH(dataset)
                cols = dataset.columns
                minus += num
            if g == 'Psychia' and self.combine_mode is not None:
                dataset, num = self.combine_Psychia(dataset)
                cols = dataset.columns
                minus += num
            if g == 'MOCA':
                dataset, num = self.combine_MOCA(dataset, edu=data['EDUC'])
                cols = dataset.columns
                minus += num

            if g == 'Info':
                add = [kk for kk in self.head if kk != 'SubID']
                cols += add
                DataAll = pd.concat([DataAll, data[add], dataset], axis=1)
            else:
                DataAll = pd.concat([DataAll, dataset], axis=1)

            Note[g] = cols

        missF, missS = count_missing_ratio(DataAll)
        F, S = [int(kk * self.missing_threshold) for kk in DataAll.shape]
        missF = {kk: vv for kk, vv in missF.items() if vv > F}
        missS = {kk: vv for kk, vv in missS.items() if vv > S}

        original_v = sum(self.Variables['Group'].value_counts())
        print(f'Due to combination, feature number {original_v}-{minus}={original_v - minus}')
        # assert original_v - minus == DataAll.shape[1] - 2
        print(DataAll.shape[1])

        print('\n' + '-' * 10 + ' Summary ' + '-' * 10)
        print(' ' * 9 + '\tOrigin\tDrop\tRemain')
        print(f'Variables\t{DataAll.shape[1]}\t{len(missF)}\t{DataAll.shape[1] - len(missF)}')
        print(f'Samples  \t{DataAll.shape[0]}\t{len(missS)}\t{DataAll.shape[0] - len(missS)}')
        print('Final missing features: ', missF)
        print('Final missing samples: ', missS)
        data_dropNaN = DataAll.drop(missS, axis=0).drop(missF, axis=1)
        print(f'Data shape after drop missing ({self.missing_threshold}): {data_dropNaN.shape}')
        assert data_dropNaN.shape[0] == len(DataAll) - len(missS) and data_dropNaN.shape[1] == original_v - minus + len(self.head) - 1
        data_dropNaN.to_csv(os.path.join(self.Save_path, f'DN{self.missing_threshold}_{self.mri}_Data.csv'), index=True)

        drop_v = sum([len(kk[1]) for kk in Note.values()])
        remain_v = original_v - drop_v
        original_s = len(data)
        drop_s = sum([len(kk[2]) for kk in Note.values()])
        remain_s = original_s - drop_s
        print(' ' * 9 + '\tOrigin\tDrop\tRemain')
        print(f'Variables\t{original_v}\t{drop_v}\t{remain_v}')
        print(f'Samples  \t{original_s}\t{drop_s}\t{remain_s}')

        stats_scope = pd.DataFrame(columns=['Group', 'Variable name', 'Range', 'Type', 'Min', 'Max'])
        for g in Note.keys():
            cols = [kk for kk in Note[g] if kk not in missF]
            for col in cols:
                values = data_dropNaN[col].values
                col_range, col_type, Min, Max = check_variables(values, col)
                stats_scope.loc[len(stats_scope)] = [g, col, col_range, col_type, Min, Max]

        stats_scope1 = pd.merge(stats_scope,
                                self.Variables[
                                    ['Group', 'Variable name', 'Form', 'Short descriptor', 'AllowableCodes']],
                                how='left')
        pd.testing.assert_frame_equal(stats_scope1[['Group', 'Variable name']], stats_scope[['Group', 'Variable name']])
        assert len(stats_scope1) == data_dropNaN.shape[1]
        stats_scope1.to_csv(os.path.join(self.Save_path, f'DN{self.missing_threshold}_{self.mri}_Stats.csv'),
                            index=False)

    def convert_missing(self, data):
        print(f"Data shape: {data.shape}")

        MissingTags = pd.Series(self.Variables['MissingTag'].values, index=self.Variables['Variable name'])
        print('\n' + '-' * 10 + ' Missing condition for every group ' + '-' * 10)
        data.index = data['SubID']
        DataAll = pd.DataFrame(index=data.index)
        Note = dict()
        minus = 0
        for g, c in self.Variables['Group'].value_counts().items():
            print(f'Group: {g}, Variable number: {c}')

            cols = self.Variables[self.Variables['Group'] == g]['Variable name'].tolist()
            dataset = data.loc[:, cols]
            for cc in cols:
                dataset[cc] = replace_missing(MissingTags[cc], dataset[cc])

            if g == 'MedH' and self.combine_mode is not None:
                dataset, num = self.combine_MedH(dataset)
                cols = dataset.columns
                minus += num
            if g == 'Psychia' and self.combine_mode is not None:
                dataset, num = self.combine_Psychia(dataset)
                cols = dataset.columns
                minus += num
            if g == 'MOCA':
                dataset, num = self.combine_MOCA(dataset, edu=data['EDUC'])
                cols = dataset.columns
                minus += num

            missF, missS = count_missing_ratio(dataset)
            F, S = [int(kk * self.missing_threshold) for kk in data.shape]
            missF = {kk: vv for kk, vv in missF.items() if vv > F}
            missS = {kk: vv for kk, vv in missS.items() if vv > S}

            if g == 'Info':
                add = [kk for kk in self.head if kk != 'SubID']
                data['NACCID'] = [int(kk.split('NACC')[1]) for kk in data['NACCID']]
                cols += add
                DataAll = pd.concat([DataAll, data[add], dataset], axis=1)
            else:
                DataAll = pd.concat([DataAll, dataset], axis=1)

            print(cols)
            Note[g] = [cols, missF, missS]
            print(f'Missing variables ({len(missF)}):\n' + str(list(missF.keys())))
            print(f'Missing samples ({len(missS)}):\n' + str(list(missS.keys())))

        print('\n' + '-' * 10 + ' Summary ' + '-' * 10)
        original_v = sum(self.Variables['Group'].value_counts())
        print(f'Due to combination, feature number {original_v}-{minus}={original_v-minus}')
        original_v = original_v - minus
        drop_v = sum([len(kk[1]) for kk in Note.values()])
        remain_v = original_v - drop_v
        original_s = len(data)
        drop_s = sum([len(kk[2]) for kk in Note.values()])
        remain_s = original_s - drop_s
        print(' ' * 9 + '\tOrigin\tDrop\tRemain')
        print(f'Variables\t{original_v}\t{drop_v}\t{remain_v}')
        print(f'Samples  \t{original_s}\t{drop_s}\t{remain_s}')

        missF = [cc for kk in Note.values() for cc in kk[1].keys()]
        missS = [cc for kk in Note.values() for cc in kk[2].keys()]
        print('Final missing features: ', missF)
        print('Final missing samples: ', missS)
        data_dropNaN = DataAll.drop(missS, axis=0).drop(missF, axis=1)
        print(f'Data shape after drop missing ({self.missing_threshold}): {data_dropNaN.shape}')
        assert data_dropNaN.shape[0] == remain_s and data_dropNaN.shape[1] == remain_v + len(self.head) - 1
        data_dropNaN.to_csv(os.path.join(self.Save_path, f'DN{self.missing_threshold}_{self.mri}_Data.csv'), index=True)

        stats_scope = pd.DataFrame(columns=['Group', 'Variable name', 'Range', 'Type', 'Min', 'Max'])
        for g in Note.keys():
            cols = [kk for kk in Note[g][0] if kk not in Note[g][1].keys()]  # 该模态下，缺失率小于阈值的特征
            for col in cols:
                values = data_dropNaN[col].values
                col_range, col_type, Min, Max = check_variables(values, col)
                stats_scope.loc[len(stats_scope)] = [g, col, col_range, col_type, Min, Max]

        stats_scope1 = pd.merge(stats_scope,
                                self.Variables[
                                    ['Group', 'Variable name', 'Form', 'Short descriptor', 'AllowableCodes']],
                                how='left')
        pd.testing.assert_frame_equal(stats_scope1[['Group', 'Variable name']], stats_scope[['Group', 'Variable name']])
        assert len(stats_scope1) == data_dropNaN.shape[1]
        stats_scope1.to_csv(os.path.join(self.Save_path, f'DN{self.missing_threshold}_{self.mri}_Stats.csv'), index=False)


    def combine_MedH(self, dataset0):  # -3-6=-9
        dataset = dataset0.copy()
        num = dataset.shape[1]
        change_dict = {
            'TBI': ['TRAUMBRF', 'TRAUMEXT', 'TRAUMCHR'],
            'PSYCDIS': ['NPSYDEV', 'OCD', 'ANXIETY', 'SCHIZ', 'BIPOLAR', 'PTSD'],
            }

        for mainCol in change_dict.keys():
            attachCol = change_dict[mainCol]
            if [kk in dataset.columns for kk in [mainCol] + attachCol] != [True] * (1 + len(attachCol)):
                continue
            print(f'combine {attachCol} to {mainCol}')
            print(f'feature number minus {len(attachCol)}')
            # print(dataset.loc[list(dataset.index)[35:45], [mainCol]+attachCol])
            c0 = combine_main_attach(dataset, mainCol, attachCol, 0)  # attach都是0
            c1 = combine_main_attach(dataset, mainCol, attachCol, 1)  # main+attach有1
            c2 = combine_main_attach(dataset, mainCol, attachCol, 2)  # main+attach有2
            # 0 = Absent; 1 = Recent/Active; 2 = Remote/Inactive
            combined = 1 * c1 + 2 * c2 * (~c1) + 0 * c0 * (~c1 * ~c2)
            combined[~c1 * ~c2 * ~c0] = dataset[mainCol]
            if self.combine_mode == 'replace':
                dataset[mainCol] = combined
                dataset.drop(attachCol, axis=1, inplace=True)
            else:
                raise ValueError('combine_mode must in [replace]')

        return dataset, num - dataset.shape[1]

    def combine_Psychia(self, dataset0):  # -12
        dataset = dataset0.copy()
        num = dataset.shape[1]
        print(f"feature number minus {len([ll for ll in dataset.columns if ll.endswith('SEV')])}")
        for mainCol in dataset.columns:
            if not mainCol.endswith('SEV'):
                continue
            attachCol = mainCol.split('SEV')[0]
            assert ((dataset[attachCol] == 0) == (dataset[mainCol] == 8)).all()
            assert [kk in [1, 2, 3, 8] for kk in dataset[mainCol]]
            assert [kk in [0, 1] for kk in dataset[attachCol]]
            # attachCol ["xxxx" in the last month]:
            # 0 = No
            # 1 = Yes
            # mainCol ["xxxx" severity]
            # 1 = Mild (noticeable, but not a significant change)
            # 2 = Moderate (significant, but not a dramatic change)
            # 3 = Severe (very marked or prominent; a dramatic change)
            # 8 = Not applicable, no delusions reported

            loc = (dataset[mainCol] != 8) & (dataset[attachCol] == 1)
            combined = dataset[mainCol] * loc + 0 * (dataset[attachCol] == 0)
            combined[(dataset[mainCol] == 8) & (dataset[attachCol] == 1)] = np.nan
            if self.combine_mode == 'replace':
                dataset[mainCol] = combined
                dataset.drop(attachCol, axis=1, inplace=True)
            else:
                raise ValueError('combine_mode must in [replace]')

        return dataset, num - dataset.shape[1]

    def combine_MOCA(self, dataset0, edu):  # -5
        dataset = dataset0.copy()
        num = dataset.shape[1]
        total = dataset['MOCATOTS'].tolist()
        correct = dataset['NACCMOCA'].tolist()
        moca_cols = dataset.drop(['MOCATOTS', 'NACCMOCA', 'MOCAREGI', 'MOCARECC', 'MOCARECR'], axis=1)
        cal_total, cal_correct = [], []
        for i in range(len(dataset)):
            if moca_cols.iloc[i].isna().sum() > 0:
                cal_total.append(np.nan)
                cal_correct.append(np.nan)
                continue
            cal_total.append(sum(moca_cols.iloc[i]))
            if not np.isnan(total[i]):
                assert total[i] == cal_total[i]

            #  If the subject has 12 years of education or fewer, a point is added to his/her total score.
            if np.isnan(edu.iloc[i]):
                cal_correct.append(np.nan)
                continue
            elif 0 <= edu.iloc[i] <= 12 and cal_total[i] < 30:
                cal_correct.append(cal_total[i] + 1)
            elif edu.iloc[i] > 12 or cal_total[i] == 30:
                cal_correct.append(cal_total[i])
            else:
                raise ValueError
            if not np.isnan(correct[i]):
                assert correct[i] == cal_correct[i]

        dataset['MOCATOTS'] = cal_total
        dataset['NACCMOCA'] = cal_correct
        return dataset, num - dataset.shape[1]

    def final_save(self, data_all, stats):
        modals = dict()
        for g, c in stats.groupby(['Group']).groups.items():
            modals[g] = data_all[stats.loc[c, 'Variable name'].tolist()]

        # save as pkl with csv
        with open(os.path.join(self.Save_path, f'DN{self.missing_threshold}_{self.mri}_csv.pkl'), 'wb') as f:
            pickle.dump(data_all, f)
        # save as pkl with 2 csv
        modal_feature = {g: stats.loc[c, 'Variable name'].tolist() for g, c in stats.groupby(['Group']).groups.items()}
        with open(os.path.join(self.Save_path, f'DN{self.missing_threshold}_{self.mri}_csv_dict.pkl'), 'wb') as f:
            pickle.dump([data_all, modal_feature], f)

        # save as split pkl
        for g in modals.keys():
            modals[g].to_pickle(os.path.join(self.Save_path, f'DN{self.missing_threshold}_{self.mri}_{g}.pkl'))
        # save as merge pkl
        with open(os.path.join(self.Save_path, f'DN{self.missing_threshold}_{self.mri}_Data.pkl'), 'wb') as f:
            pickle.dump(modals, f)

# ############# functions #############


def replace_missing(tag, series0):
    tag = str(tag)
    convert, flag = convert_float(series0.values)

    if flag:
        series, _ = convert_categorical(convert)
    else:
        assert tag in ['', 'nan']
        return convert

    if tag in ['', 'nan']:
        pass
    elif tag == '; ':
        assert flag
        digit = np.ceil(np.log10(np.nanmax(np.array(series))))
        MAX = int('8' * int(digit))
        series = [(kk if kk < MAX else np.nan) for kk in series]
    else:
        assert flag
        replace = [kk.replace(' ', '') for kk in tag.split(';')]
        replace = [int(kk) for kk in replace if kk != '']
        series = [(kk if kk not in replace else np.nan) for kk in series]
    return series


def combine_main_attach(dataset, mainCol, attachCol, value):
    if np.isnan(value):
        a = [True] * len(dataset)
        for col in attachCol:
            a *= pd.isna(dataset[col])
    elif value == 0:
        a = [True] * len(dataset)
        for col in attachCol:
            a *= dataset[col] == value
    else:
        a = dataset[mainCol] == value
        for col in attachCol:
            a += dataset[col] == value
    return a


def count_missing_ratio(data):
    missF = pd.isna(data).sum(axis=0)
    missS = pd.isna(data).sum(axis=1)
    return missF.to_dict(), missS.to_dict()


def check_variables(values, col=None):
    col_values, is_float = convert_float(values)
    if is_float:
        Min, Max = np.nanmin(col_values), np.nanmax(col_values)
        col_set = set([kk for kk in col_values if not np.isnan(kk)])
        col_values, is_categorical = convert_categorical(col_set)
        if is_categorical and len(col_set) < 8:
            col_set = set(col_values)
            col_range, col_type = sorted(col_set), 'category (int)'
        elif is_categorical and len(col_set) >= 8:
            # print(f'{g},{col},{int(Min):d}~{int(Max):d},continuous (int)')
            col_range, col_type = f'{int(Min):d}~{int(Max):d}', 'continuous (int)'
        else:
            # print(f'{g},{col},{Min:.3f}~{Max:.3f},continuous (float)')
            col_range, col_type = f'{Min:.3f}~{Max:.3f}', 'continuous (float)'
    else:  # is str
        col_set = set([kk for kk in col_values if kk != ''])
        Min, Max = -999, -999
        if len(col_set) <= 10:
            assert col in ['Label', 'PACKET']
            # print(f'{g},{col},{Range},category (str)')
            col_range, col_type = sorted(col_set), 'category (str)'
        else:
            assert col in ['NACCID', 'NACCMRFI']
            # print(f'{g},{col},SKIP,None (str)')
            col_range, col_type = 'SKIP', 'None (str)'
    return [col_range, col_type, Min, Max]


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
