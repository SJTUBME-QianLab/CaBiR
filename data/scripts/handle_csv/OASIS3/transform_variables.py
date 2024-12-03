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


class TransformVariables:
    def __init__(self):
        # self.RawTable_path = r"D:\PublicData\OASIS\files\OASIS3_data_files\scans"
        self.RawTable_path = "/home/data2/PublicData/OASIS/files/OASIS3_data_files/scans"

        self.Save_path = os.getcwd().split('scripts')[0] + '/handle_csv/OASIS3/'  # handled csvs will be saved here
        self.save_log = True
        self.combine_mode = 'replace'  # [replace, modify, add, drop]
        # replace：取代DELSEV，只有DELSEV；modify：取代DELSEV，有DELSEV和DEL；
        # add：增加myDEL，保留原来的DELSEV和DEL；drop：增加myDEL，只有myDEL

        self.backup()

        self.img = pd.read_csv(os.path.join(self.Save_path, "mri_path_add.csv"))
        self.img.rename(columns={'subject_id': 'OASISID'}, inplace=True)
        self.img.insert(
            loc=self.img.columns.get_loc('OASISID') + 1, column='days_to_visit0',
            value=self.img['link'].apply(lambda x: int(x.split('.')[1].split('d')[1]))
        )

        self.data = self.add_label()
        # self.data = pd.read_csv('./tmp.csv', index_col=0)
        self.subj2Row = self.data.groupby(['OASISID']).groups

    def backup(self):
        timestamp = time.strftime('%Y%m%d%H%M%S')
        shutil.copy2('transform_variables.py', os.path.join(os.getcwd(), 'backup', f"transform_variables_{timestamp}.py"))
        if self.save_log:
            sys.stdout = Logger("transform", timestamp)
        print(f"Parameters:\nCombine mode: {self.combine_mode}\n")

    def transform(self):
        print('\n' + '#' * 10 + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' Transform Variables ' + '#' * 10)
        self.add_Demo()  # Demo
        self.add_APOE()  # Demo
        self.add_FAQ()  # Func
        self.add_NACCFAM()  # MedH
        self.add_MedH()  # MedH
        self.add_NPIQ()  # Psychia
        self.add_CDR_MMSE()  # Psycho
        self.add_NACCGDS()  # Psycho
        self.add_pychometrics()  # Psycho
        # self.data.to_csv('./tmp.csv', index=True)

    def get_csv(self, package, file):
        path = os.path.join(self.RawTable_path, package, 'resources', 'csv', 'files', file)
        assert os.path.isfile(path)
        return pd.read_csv(path)

    def pre_merge(self):
        print('\n' + '#' * 10 + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' Merge with MRI files ' + '#' * 10)

        # 合并脑影像信息和表格信息
        base_cols = ['OASISID', 'days_to_visit0', 'label',
                     'filename', 'link', 'path', 'shape', 'resolX', 'resolY', 'resolZ',
                     'MagneticFieldStrength', 'Manufacturer', 'ManufacturersModelName', 'SliceThickness',
                     'SeriesDescription', 'ScanningSequence']
        dataAdd = pd.merge(self.img[base_cols], self.data, on=base_cols[:3], how='left')
        print(f"Number of matched samples: {len(dataAdd.dropna(subset=['Label']))} (with replicates)")  # 1016
        assert pd.isna(dataAdd['link']).sum() == 0

        # 核对关键信息，网页爬取的MRI信息转换为变量值
        dataAdd = check(dataAdd)

        # 添加被试编号
        dataAdd.insert(loc=0, column='SubID', value='')
        for i in dataAdd.index:
            dataAdd.loc[i, 'SubID'] = dataAdd.loc[i, 'link'].split('.nii')[0]
        assert dataAdd['SubID'].nunique() == len(dataAdd)

        non_table = dataAdd[dataAdd['Label'].isna()].shape[0]
        print(f'{non_table} samples (in mri_path_add.csv) have no matched table records (in Transformed_Data.csv).')
        # 4115 - 1606 = 2509

        unsatisfied = pd.read_excel('/home/data2/PublicData/OASIS/OASIS3_DataWash.xlsx', sheet_name='mri_path_add')
        unsatisfied = unsatisfied[unsatisfied['Use'] == 2]  # 层厚、层数不满足要求的样本
        path1 = '/home/data2/MRIProcess/OASIS3/nc_MNI_brain'
        path2 = '/home/data2/MRIProcess/OASIS3/drop'
        dataAdd['Use'] = np.nan
        for i in dataAdd.index:
            if dataAdd.loc[i, 'link'] in unsatisfied['link'].values:
                continue
            elif not (os.path.isfile(os.path.join(path1, dataAdd.loc[i, 'link'])) or
                      os.path.isfile(os.path.join(path2, dataAdd.loc[i, 'link']))):
                # print(dataAdd.loc[i, 'link'])
                dataAdd.loc[i, 'Use'] = 3  # 预处理过程中出现问题，导致不存在对应的nii.gz文件
            elif isNaN(dataAdd.loc[i, 'Label']) and dataAdd.loc[i, 'Use'] != 1:
                dataAdd.loc[i, 'Use'] = 0  # 没有对应的表格数据
            else:
                dataAdd.loc[i, 'Use'] = 1  # 满足要求

        count0 = dataAdd[dataAdd['Use'] == 0].shape[0]
        count1 = dataAdd[dataAdd['Use'] == 1].shape[0]
        count3 = dataAdd[dataAdd['Use'] == 3].shape[0]
        print(f'{len(unsatisfied)} samples (in mri_path_add.csv) have unsatisfied thick or size')
        print(f'{count3} samples (in mri_path_add.csv) have fail in preprocessing')
        print(f'Without Use=2 or 3, {count0} samples (in mri_path_add.csv) have no matched table records')
        print(f'Finally, {count1} samples are available for further analysis.')
        len_mni = len([kk for kk in (os.listdir(path1) + os.listdir(path2)) if kk.endswith('.nii.gz')])
        print(f'{len(dataAdd)} = {len_mni} (MNI_brain) + '
              f'{len(unsatisfied)} (unsatisfied for thick or size) + {count3} (fail in preprocessing)')
        assert len_mni + len(unsatisfied) + count3 == len(dataAdd)
        dataAdd.to_csv(os.path.join(self.Save_path, 'Merged_Data.csv'), index=False)

    def add_label(self):
        table = self.get_csv('UDSd1-Form_D1__Clinician_Diagnosis___Cognitive_Status_and_Dementia', 'OASIS3_UDSd1_diagnoses.csv')
        for i in table.index:
            assert table.loc[i, 'OASIS_session_label'] == f"{table.loc[i, 'OASISID']}_UDSd1_d{table.loc[i, 'days_to_visit']:04d}"

        # 1-Yes | 0-No; 1-Present | 0-Absent
        NC_cols = ['NORMCOG']
        # 2. Does the subject have normal cognition (no MCI, demential, or other neurological condition
        # resulting in cognitive impairment)?
        MCI_cols = [
            'MCIAMEM',  # 4a. Amnestic MCI-memory impairment only
            'MCIAPLUS',  # 4b. Amnestic MCI-memory impairment plus one or more other domains
            'MCIAPLAN',  # 4b1. Language
            'MCIAPATT',  # 4b2. Attention
            'MCIAPEX',  # 4b3. Executive function
            'MCIAPVIS',  # 4b4. Visuospatial
            # 4c-4d. Non-amnestic MCI ... 非遗忘型，不算在内
            # 4e. Impaired, not MCI ... 不算在内
        ]
        AD_cols = [
            'PROBAD',  # 5. Probable AD
            'POSSAD',  # 6. Possible AD
        ]
        EX_cols = [
            # Exclude others
            'DLB', 'VASC', 'VASCPS', 'ALCDEM', 'DEMUN', 'FTD', 'PPAPH', 'PNAPH', 'SEMDEMAN', 'SEMDEMAG', 'PPAOTHR',
            'PSP', 'CORT', 'HUNT', 'PRION', 'MEDS', 'DYSILL', 'DEP', 'OTHPSY', 'DOWNS', 'PARK', 'STROKE', 'HYCEPH',
            'BRNINJ', 'NEOP', 'COGOTH', 'COGOTH2', 'COGOTH3'
        ]
        str_values = {1: [1], 0: [0, np.nan]}
        all_cols = ['OASISID', 'days_to_visit', 'DEMENTED'] + NC_cols + MCI_cols + AD_cols + EX_cols

        data = []
        for i, row in table.iterrows():
            if row['OASISID'] not in self.img['OASISID'].unique():
                continue
            row = row[all_cols]
            ex_cols = find_all_indexes(EX_cols, row, str_values, target=1)
            nc_cols = find_all_indexes(NC_cols, row, str_values, target=1)
            mci_cols = find_all_indexes(MCI_cols, row, str_values, target=1)
            ad_cols = find_all_indexes(AD_cols, row, str_values, target=1)
            dementia = search_value(str_values, row['DEMENTED'])
            # if len(ex_cols) > 0:
            #     # print(f"Exclude {target['link']} because {ex_cols}")
            #     if not (len(nc_cols) + len(mci_cols) + len(ad_cols) == 0 and dementia != 0):
            #         print(ex_cols, nc_cols, mci_cols, ad_cols, dementia)
            #     continue
            print_sub = print_format(row[['OASISID', 'days_to_visit']])
            if len(nc_cols) > 0:
                if not (len(mci_cols) + len(ad_cols) == 0 and dementia != 1):
                    print(f"Conflict NC {print_sub}\t", ex_cols, nc_cols, mci_cols, ad_cols, dementia)
                    continue
                row['Label'] = 'NC'
            elif len(mci_cols) > 0:
                if not (len(nc_cols) == 0 and dementia != 1):  # 有可能len(ad_cols)>0
                    print(f"Conflict MCI {print_sub}\t", ex_cols, nc_cols, mci_cols, ad_cols, dementia)
                    continue
                row['Label'] = 'MCI'
            elif len(ad_cols) > 0 and dementia == 1:
                if not (len(nc_cols) + len(mci_cols) == 0):
                    print(f"Conflict AD {print_sub}\t", ex_cols, nc_cols, mci_cols, ad_cols, dementia)
                    continue
                row['Label'] = 'AD'
            else:
                # print("SKIP", ex_cols, nc_cols, mci_cols, ad_cols, dementia)
                continue
            data.append(row)
        data = pd.DataFrame(data)
        data = data.sort_values(by=['OASISID', 'days_to_visit'], inplace=False).reset_index(drop=True)
        print(f"Number of tabular samples: {len(data)}")  # 6715

        dataAdd = match_table(self.img, data)
        dataAdd = dataAdd.dropna(subset=['days_to_visit']).reset_index(drop=True)
        print(f"Number of MRI samples: {len(self.img)}, drop replicates: {self.img['label'].nunique()}")  # 4115, 2831
        print(f"Number of matched samples: {len(dataAdd)} (without replicates)")  # 1200

        return dataAdd

    def catch_value(self, table, NACC2ADNI, merge_cols=None):
        if merge_cols is None:
            merge_cols = ['OASISID', 'days_to_visit']
            data = pd.merge(self.data, table[merge_cols + [kk[0] for kk in NACC2ADNI.values()]],
                            how='left', on=merge_cols)
            for nacc_col, (adni_col, convert) in NACC2ADNI.items():
                for i in self.data.index:
                    self.data.loc[i, nacc_col] = search_value(convert, data.loc[i, adni_col])

            data = pd.merge(self.data[['OASISID']], table, how='left', on=['OASISID']).drop_duplicates()
            data2Row = data.groupby(['OASISID']).groups
            for ID, rows in self.subj2Row.items():
                if ID not in data2Row.keys():
                    continue
                subj = data.loc[data2Row[ID], :]
                for ii in rows:
                    target = self.data.loc[ii, :]
                    idx, _ = find_closest(target, subj)
                    if idx is None:
                        continue
                    row = data.loc[data2Row[ID][idx], :]
                    for nacc_col, (adni_col, convert) in NACC2ADNI.items():
                        if isNaN(target[nacc_col]):
                            self.data.loc[ii, nacc_col] = search_value(convert, row[adni_col])

        else:
            data = pd.merge(self.data, table[merge_cols + [kk[0] for kk in NACC2ADNI.values()]],
                            how='left', on=merge_cols)
            for nacc_col, (adni_col, convert) in NACC2ADNI.items():
                for i in self.data.index:
                    self.data.loc[i, nacc_col] = search_value(convert, data.loc[i, adni_col])

    def add_Demo(self):
        table = self.get_csv('demo-demographics', 'OASIS3_demographics.csv')
        NACC2ADNI = {
            'SEX': ['GENDER', {1: [1], 2: [2]}],
            # 1-Male 2- Female
            'RACE': ['race', {1: ['White'], 2: ['Black'], 3: ['AIAN'], 5: ['ASIAN']}],
            # NACC: 1=White; 2=Black or African American; 3=American Indian or Alaska Native;
            # 4=Native Hawaiian or Other Pacific Islander; 5=Asian; 50=Other (specify); 99=Unknown
            'EDUC': ['EDUC', {'same': np.arange(0, 30).astype(int)}],
            'HANDED': ['HAND', {1: ['L'], 2: ['R'], 3: ['B']}],
            # NACC: 1=Left; 2=Right; 3=Ambidextrous; 9=Unknown
            # OASIS: R-Right | L-Left |B-Ambidextrous
        }
        self.catch_value(table, NACC2ADNI, merge_cols=['OASISID'])

        data = pd.merge(self.data, table[['OASISID', 'AgeatEntry']],
                        how='left', on=['OASISID'])
        for i in self.data.index:
            self.data.loc[i, 'NACCAGE'] = cal_age(data.loc[i, 'days_to_visit0'], data.loc[i, 'AgeatEntry'])

        table = self.get_csv('UDSa1-Form_A1__Subject_Demographics', 'OASIS3_UDSa1_participant_demo.csv')
        convert = {'same': np.arange(1, 7).astype(int)}
        # NACC: 1=Married; 2=Widowed; 3=Divorced; 4=Separated;
        # 5=Never Married (or marriage was annulled); 6=Living as married/domestic partner; 9=Unknown
        # 1-Married | 2-Widowed | 3-Divorced | 4-Separated |
        # 5-Never Married | 6 -Living as married | 8-Other (specify) | 9-Unknown
        self.data['MARISTAT'] = np.nan
        for ptid in self.subj2Row.keys():
            sub = self.data[self.data['OASISID'] == ptid]
            tab = table[table['OASISID'] == ptid]
            values = [search_value(convert, kk) for kk in tab['MARISTAT']]
            values = np.unique([kk for kk in values if not isNaN(kk)])
            if len(values) == 0:
                continue
            elif len(values) == 1:
                value = values[0]
                self.data.loc[sub.index, 'MARISTAT'] = [value] * len(sub)
                continue
            else:
                pass
                # print(f"{ptid} MARISTAT inconsistent: \n{tab[['days_to_visit', 'MARISTAT']].dropna()}")
            final = np.nan
            for id_t in tab.index:
                tmp = search_value(convert, tab.loc[id_t, 'MARISTAT'])
                tmp_date = tab.loc[id_t, 'days_to_visit']
                if not isNaN(tmp) and isNaN(final):  # 初次遇到实际值，则把之前的空值都填上
                    for id_s in sub.index:
                        if sub.loc[id_s, 'days_to_visit0'] - tmp_date <= 90:
                            # 如果此行时间早于tmp_date+90天，则其值设为当前值tmp
                            self.data.loc[id_s, 'MARISTAT'] = tmp
                    final = tmp
                elif not isNaN(tmp) and not isNaN(final):  # 再次遇到实际值，则最近的填tmp、之前的填update，再更新update
                    for id_s in sub.index:
                        if abs(sub.loc[id_s, 'days_to_visit0'] - tmp_date) <= 90:
                            # 如果此行时间与tmp_date前后相差90天内，则其值设为当前值tmp
                            self.data.loc[id_s, 'MARISTAT'] = tmp
                        elif (isNaN(self.data.loc[id_s, 'MARISTAT']) and
                                sub.loc[id_s, 'days_to_visit0'] - tmp_date <= 90):
                            # 如果此行值为空，且时间早于tmp_date+90天，则其值设为历史值final
                            self.data.loc[id_s, 'MARISTAT'] = final
                    final = tmp

            for id_s in sub.index:  # 末尾填上final
                if isNaN(self.data.loc[id_s, 'MARISTAT']):
                    self.data.loc[id_s, 'MARISTAT'] = final

    def add_APOE(self):
        table = self.get_csv('demo-demographics', 'OASIS3_demographics.csv')
        NACC2ADNI = {
            'NACCAPOE': ['APOE', {1: [33], 2: [34], 3: [23], 4: [44], 5: [24], 6: [22]}],
            # NACC: 1=e3,e3; 2=e3,e4; 3=e3,e2; 4=e4,e4; 5=e4,e2; 6=e2,e2; 9=Missing/ unknown/ not assessed
            'NACCNE4S': ['APOE', {0: [22, 23, 33], 1: [24, 34], 2: [44]}],
            # NACC: 0=No e4 allele, 1=1 copy of e4 allele, 2=2 copies of e4 allele, 9=Missing/ unknown/ not assessed"
        }
        data = pd.merge(self.data, table[['OASISID', 'APOE']],
                        how='left', on=['OASISID'])
        for nacc_col, (_, convert) in NACC2ADNI.items():
            for i in self.data.index:
                self.data.loc[i, nacc_col] = search_value(convert, data.loc[i, 'APOE'])

    def add_FAQ(self):
        table = self.get_csv('UDSb7-Form_B7__Functional_Assessment___FAQ', 'OASIS3_UDSb7_faq_fas.csv')
        common = {'same': [0, 1, 2, 3]}
        # NACC: 0=Normal; 1=Has difficulty, but does by self; 2=Requires assistance; 3=Dependent;
        # 8=Not applicable (e.g., never did); 9=Unknown; -4=Not available
        # OASIS: 0-Normal | 1-Has difficulty, but does by self | 2-Requires assistance | 3-Dependent |
        # 8-Not applicable | 9-Unknown
        col_names = [
            'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE',
            'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL',
        ]
        NACC2ADNI = {nn: [nn, common] for nn in col_names}
        self.catch_value(table, NACC2ADNI)

    def add_NACCFAM(self):
        table = self.get_csv('UDSa3-Form_A3__Subject_Family_History', 'OASIS3_UDSa3.csv')
        cols = [kk for kk in table.columns if ('DEM' in kk and kk != 'RELSDEM')]
        str_values = {1: [1], 0: [0, np.nan], np.nan: [9]}
        # NACC: 0=No report of a first-degree family member with cognitive impairment
        # 1=Report of at least one first-degree ...; 9=Unknown; -4=Not available
        # OASIS: 0-No | 1-Yes | 9-Unknown

        merge_cols = ['OASISID', 'days_to_visit']
        data = pd.merge(self.data, table[merge_cols + cols],
                        how='left', on=merge_cols)
        assert (data.index == self.data.index).all()
        for i in self.data.index:
            row = data.loc[i, :]
            values = [search_value(str_values, row[kk]) for kk in cols]
            if all(isNaN(kk) for kk in values):
                self.data.loc[i, 'NACCFAM'] = np.nan
            else:
                self.data.loc[i, 'NACCFAM'] = int(np.nansum(values) > 0)

        data = pd.merge(self.data[['OASISID']], table[merge_cols + cols], how='left', on=['OASISID']).drop_duplicates()
        data2Row = data.groupby(['OASISID']).groups
        for ID, rows in self.subj2Row.items():
            if ID not in data2Row.keys():
                continue
            subj = data.loc[data2Row[ID], :]
            for ii in rows:
                target = self.data.loc[ii, :]
                idx, _ = find_closest(target, subj)
                if idx is None:
                    continue
                row = data.loc[data2Row[ID][idx], :]
                if isNaN(target['NACCFAM']):
                    values = [search_value(str_values, row[kk]) for kk in cols]
                    if all(isNaN(kk) for kk in values):
                        self.data.loc[ii, 'NACCFAM'] = np.nan
                    else:
                        self.data.loc[ii, 'NACCFAM'] = int(np.nansum(values) > 0)

    def add_MedH(self):
        table = self.get_csv('UDSa5-Form_A5__Subject_Health_History', 'OASIS3_UDSa5_health_history.csv')
        NACC2ADNI = {
            'SMOKYRS': ['SMOKYRS', {'same': np.arange(1, 100).astype(int)}],
            # NACC: Total years smoked cigarettes.
            # 0 - 87; 88=Not applicable; 99=Unknown; -4=Not available
            'PACKSPER': ['PACKSPER', {'same': np.arange(0, 6).astype(int)}],
            # NACC: Average number of packs smoked per day.
            # 0=No reported cigarette use; 1=1 cigarette to less than 1/2 pack; 2=1/2 pack to less than 1 pack;
            # 3=1 pack to 11/2  packs; 4=11/2  packs to 2 packs; 5=More than two packs;
            # 8=Not applicable; 9=Unknown; -4=Not available
            'DEPOTHR': ['DEPOTHR', {'same': [0, 1]}],
            # NACC: Depression episodes more than two years ago
            # 0=No; 1=Yes; 9=Unknown; -4=Not available
        }
        NACC2ADNI.update({kk: [kk, {'same': [0, 1, 2]}] for kk in
                          ['CVHATT', 'CBSTROKE', 'HYPERTEN', 'ALCOHOL', 'PSYCDIS']})
        self.catch_value(table, NACC2ADNI)

    def add_NPIQ(self):
        table = self.get_csv('UDSb5-Form_B5__Behavioral_Assessment__NPI_Q', 'OASIS3_UDSb5_npiq.csv')
        cols = ['DEL', 'HALL', 'AGIT', 'DEPD', 'ANX', 'ELAT', 'APA', 'DISN', 'IRR', 'MOT', 'NITE', 'APP']
        # DEL ["xxxx" in the last month]:
        # 0 = No
        # 1 = Yes
        # DELSEV ["xxxx" severity]
        # 1 = Mild (noticeable, but not a significant change)
        # 2 = Moderate (significant, but not a dramatic change)
        # 3 = Severe (very marked or prominent; a dramatic change)
        # 8 = Not applicable, no delusions reported
        # OASIS:
        # DEL:      0 - No | 1 - Yes | 9 - Unknown
        # DELSEV:   1 - Mild | 2 - Moderate | 3 - Severe | 9 - Unknown

        merge_cols = ['OASISID', 'days_to_visit']
        data = pd.merge(self.data, table, how='left', on=merge_cols)
        assert (data.index == self.data.index).all()
        for ii in self.data.index:
            row = data.loc[ii, :]
            for nacc_col in cols:
                assert nacc_col in row and f'{nacc_col}SEV' in row
                yn = row[nacc_col]
                sev = row[f'{nacc_col}SEV']
                self.data.loc[ii, nacc_col] = yn
                self.data.loc[ii, f'{nacc_col}SEV'] = sev
                if yn == 1 and sev in [1, 2, 3]:
                    # 过去一个月出现过，且存在等级，则用它
                    self.data.loc[ii, f'my{nacc_col}SEV'] = sev
                elif yn == 0:  # 过去一个月没出现过，则0
                    if sev in [1, 2, 3]:
                        print(f"Conflict {row['label']} {nacc_col}: yn=0, sev={sev}")
                    self.data.loc[ii, f'my{nacc_col}SEV'] = 0
                else:  # 其他都算缺失值 nan
                    if sev in [1, 2, 3]:
                        print(f"Conflict {row['label']} {nacc_col}: yn={yn}, sev={sev}")

        data = pd.merge(self.data[['OASISID']], table, how='left', on=['OASISID']).drop_duplicates()
        data2Row = data.groupby(['OASISID']).groups
        for ID, rows in self.subj2Row.items():
            if ID not in data2Row.keys():
                continue
            subj = data.loc[data2Row[ID], :]
            for ii in rows:
                target = self.data.loc[ii, :]
                idx, _ = find_closest(target, subj)
                if idx is None:
                    continue
                row = data.loc[data2Row[ID][idx], :]
                for nacc_col in cols:
                    if isNaN(target[nacc_col]):
                        yn = row[nacc_col]
                        sev = row[f'{nacc_col}SEV']
                        self.data.loc[ii, nacc_col] = yn
                        self.data.loc[ii, f'{nacc_col}SEV'] = sev
                        if yn == 1 and sev in [1, 2, 3]:
                            # 过去一个月出现过，且存在等级，则用它
                            self.data.loc[ii, f'my{nacc_col}SEV'] = sev
                        elif yn == 0:  # 过去一个月没出现过，则0
                            if sev in [1, 2, 3]:
                                print(f"Conflict {row['label']} {nacc_col}: yn=0, sev={sev}")
                            self.data.loc[ii, f'my{nacc_col}SEV'] = 0
                        else:  # 其他都算缺失值 nan
                            if sev in [1, 2, 3]:
                                print(f"Conflict {row['label']} {nacc_col}: yn={yn}, sev={sev}")

        if self.combine_mode == 'replace':  # 取代原来的main特征，删除attach特征
            self.data[[f'{kk}SEV' for kk in cols]] = self.data[[f'my{kk}SEV' for kk in cols]]
            self.data.drop([kk for kk in cols] + [f'my{kk}SEV' for kk in cols], axis=1, inplace=True)
        elif self.combine_mode == 'modify':  # 取代原来的main特征，保留attach特征
            self.data[[f'{kk}SEV' for kk in cols]] = self.data[[f'my{kk}SEV' for kk in cols]]
            self.data.drop([f'my{kk}SEV' for kk in cols], axis=1, inplace=True)
        elif self.combine_mode == 'add':  # 新增my+main特征，保留原来的main和attach特征
            pass
        elif self.combine_mode == 'drop':  # 新增my+main特征，删除原来的main和attach特征
            self.data.drop([f'{kk}SEV' for kk in cols] + [kk for kk in cols], axis=1, inplace=True)
        else:
            raise ValueError('combine_mode must in [replace, modify, add, drop]')

    def add_CDR_MMSE(self):
        table = self.get_csv('UDSb4-Form_B4__Global_Staging__CDR__Standard_and_Supplemental', 'OASIS3_UDSb4_cdr.csv')
        NACC2ADNI = {
            'CDRGLOB': ['CDRTOT', {'same': [0, 0.5, 1, 2, 3]}],
            # NACC: Global CDR
            # 0.0 = No impairment; 0.5 = Questionable impairment; 1.0 = Mild impairment; 2.0 = Moderate impairment; 3.0 = Severe impairment
            'CDRSUM': ['CDRSUM', {'same': np.arange(0, 18.5, 0.5)}],
            # NACC: Standard CDR sum of boxes
            # 0.0, 0.5, 1.0, 1.5, ..., 18.0 (scores of 16.5 and 17.5 not possible)
            'MMSE': ['MMSE', {'same': np.arange(0, 31).astype(int)}],
            # NACC: Total MMSE score (using D-L-R-O-W)
            # 0 - 30
        }
        for i in range(len(table)):
            cdr_sum = np.sum(table.loc[i, ['memory', 'orient', 'judgment', 'commun', 'homehobb', 'perscare']])
            CDRSUM = table.loc[i, 'CDRSUM']
            if isNaN(CDRSUM) or isNaN(cdr_sum):
                continue
            if CDRSUM != cdr_sum:
                print(f"Conflict {table.loc[i, 'OASIS_session_label']} CDRSUM: {CDRSUM} != {cdr_sum}")
                table.loc[i, 'CDRSUM'] = np.nan
        self.catch_value(table, NACC2ADNI)

    def add_NACCGDS(self):
        table = self.get_csv('UDSb6-Form_B6__Behavioral_Assessment___GDS', 'OASIS3_UDSb6_gds.csv')
        NACC2ADNI = {
            'NACCGDS': ['GDS', {'same': np.arange(0, 16).astype(int)}],
            # NACC: Total GDS Score
            # 0 - 15; 88 = Could not be calculated; -4 = Not available
        }
        self.catch_value(table, NACC2ADNI)

    def add_pychometrics(self):
        table = self.get_csv('pychometrics-Form_C1__Cognitive_Assessments', 'OASIS3_UDSc1_cognitive_assessments.csv')
        NACC2ADNI = {
            'ANIMALS': ['ANIMALS', {'same': np.arange(0, 78).astype(int)}],  # 0-77
            # NACC: Animals - Total number of animals named in 60 seconds
            # OASIS3: CATEGORY FLUENCY - ANIMALS
            'TRAILA': ['TRAILA', {'same': np.arange(0, 151).astype(int)}],  # 0-150
            # NACC: Trail Making Test Part A - Total number of seconds to complete
            # OASIS3: TRAILMAKING A (number of seconds)
            'TRAILB': ['trailb', {'same': np.arange(0, 301).astype(int)}],  # 0-300
            # NACC: Trail Making Test Part B - Total number of seconds to complete
            'LOGIMEM': ['LOGIMEM', {'same': np.arange(0, 26).astype(int)}],  # 0-25
            # NACC: Total number of story units recalled from this current test administration
            # OASIS3: LOGICAL MEMORY IA – Immediate
            'MEMUNITS': ['MEMUNITS', {'same': np.arange(0, 26).astype(int)}],  # 0-25
            # NACC: Logical Memory IIA - Delayed - Total number of story units recalled
            # OASIS3: LOGICAL MEMORY IIA – Delayed
            'DIGIF': ['DIGIF', {'same': np.arange(0, 13).astype(int)}],  # 0-12
            # NACC: Digit span forward trials correct
            # OASIS3: WMS-R DIGIT SPAN FORWARD (total number of trials)
            'DIGIFLEN': ['digfor', {'same': np.arange(0, 9).astype(int)}],  # 0-8
            # NACC: Digit span forward length
            # OASIS3: WMS Digit Span Forward
            'DIGIB': ['DIGIB', {'same': np.arange(0, 13).astype(int)}],  # 0-12
            # NACC: Digit span backward trials correct
            # OASIS3: WMS-R DIGIT SPAN BACKWARD (total number of trials)
            'DIGIBLEN': ['digback', {'same': np.arange(0, 9).astype(int)}],  # 0-8
            # NACC: Digit span backward length
            # OASIS3: WMS Digit Span Backward
            'BOSTON': ['bnt', {'same': np.arange(0, 31).astype(int)}],  # 0-30
            # NACC: Boston Naming Test (30) - Total score
            # OASIS3: Boston Naming Test (60 items)
        }
        self.catch_value(table, NACC2ADNI)


def search_value(str_value, value):
    if len(str_value) == 1 and 'same' in str_value.keys():
        if value in str_value['same']:
            return value
        else:
            return np.nan
    else:
        for key in str_value.keys():
            if value in str_value[key]:
                return key
    return np.nan


def find_all_indexes(lst, row, str_values, target=1):
    find = []
    for kk in lst:
        value = search_value(str_values, row[kk])
        if value == target:
            find.append(kk)
    return find


def find_closest(target, search, target_col='days_to_visit', search_col='days_to_visit'):
    assert target_col in target and search_col in search
    min_diff = 1000000
    ans = None
    print_cols = ['OASISID', target_col, search_col]
    target_time = target[target_col]
    if isNaN(target_time):
        return None, None
    for i in range(len(search)):
        search_time = search.iloc[i][search_col]
        if isNaN(search_time):
            continue
        diff = abs(target_time - search_time)
        if diff < min_diff and diff <= 90:  # within +/- 3 months
            min_diff = diff
            ans = i
    return ans, min_diff


def cal_age(days, baseline):
    if isNaN(days) or isNaN(baseline):
        return np.nan
    return np.round(days / 365 + float(baseline), 2)


def match_table(data, Table):
    merged = data[['OASISID', 'days_to_visit0', 'label']].drop_duplicates(keep='first').reset_index(drop=True)
    assert (merged['label'].values == data['label'].unique()).all()
    # 根据 Table (mri_path_add.csv) 中的 days_to_visit 和 data 中的 days_to_visit0 进行匹配
    for i in merged.index:
        target = merged.loc[i, :]
        subj = Table[Table['OASISID'] == target['OASISID']]
        idx, _ = find_closest(target, subj,
                              target_col='days_to_visit0', search_col='days_to_visit')
        if idx is None:
            continue
        merged.loc[i, Table.columns] = subj.iloc[idx, :]
    return merged


def check(merged):
    # MagneticFieldStrength - MRIFIELD; Manufacturer - MRIMANU; Mfg Model - MRIMODL;
    manu_dict = {
        'GE MEDICAL SYSTEMS': 1,
        'Philips Medical Systems': 2,
        'SIEMENS': 3,
        'Siemens': 3,
        np.nan: np.nan
    }
    modl_dict = {
        'DISCOVERY_MR750': 1,
        'GENESIS_SIGNA': 2,
        'Signa_HDxt': 3,
        'TrioTim': 4,
        'Eclipse_1.5T': 5,
        'Allegra': 6,
        'SIGNA_EXCITE': 7,
        'SIGNA': 8, 'SIGNA_PET_MR': 8, 'SIGNA_Premier': 8, 'SIGNA_UHP': 8, 'Signa_MR360': 8,
        'GEMINI': 9,
        'Ingenuity': 10,
        'Sonata': 11,
        'Skyra': 12,
        'SIGNA_HDx': 13,
        'Achieva': 14, 'Achieva_dStream': 14,
        'Prisma': 15, 'Prisma_fit': 15,
        'Verio': 16,

        'Aera': 17,
        'Avanto': 18,
        'Biograph_mMR': 19,
        'Espree': 20,
        'Intera': 21,
        'Symphony': 22, 'SymphonyTim': 22,
        'Trio': 23,

        'SIGNA EXCITE': 7,
        'SIGNA HDx': 24,
        'Signa HDxt': 24,
        'SonataVision': 11,
        'Gyroscan Intera': 25,
        'NUMARIS/4': 26,

        'MAGNETOM_Vida': 27,
        'Vision': 11,
        np.nan: np.nan
    }

    for i in merged.index:  # 全都以网页爬取的为准
        if isNaN(merged.loc[i, 'MagneticFieldStrength']):
            merged.loc[i, 'MRIFIELDj'] = np.nan
        else:
            field = np.round(merged.loc[i, 'MagneticFieldStrength'], 1)
            assert field in [1.5, 3.0]
            merged.loc[i, 'MRIFIELDj'] = field

        merged.loc[i, 'MRIMANUj'] = manu_dict[merged.loc[i, 'Manufacturer']]

        merged.loc[i, 'MRIMODLj'] = modl_dict[merged.loc[i, 'ManufacturersModelName']]

        merged.loc[i, 'MRITHICKj'] = merged.loc[i, 'SliceThickness']

    return merged


if __name__ == '__main__':
    obj = TransformVariables()
    obj.transform()
    obj.pre_merge()
