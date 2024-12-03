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
from collect_samples import Logger, print_format, isNaN, trans_VSCODE


class TransformVariables:
    def __init__(self):
        self.RawTable_path = "PublicData/AIBL/Data_extract_3.3.0"

        self.Save_path = os.getcwd().split('scripts')[0] + '/handle_csv/AIBL/'  # handled csvs will be saved here
        self.save_log = True
        self.combine_mode = 'replace'

        self.backup()

        self.img = pd.read_csv(os.path.join(self.Save_path, "mri_path_add.csv"))
        self.img.rename(columns={'Subject': 'RID'}, inplace=True)

        self.RIDs = self.img['RID'].unique()
        self.data = self.add_label()

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
        self.add_MedH()  # MedH
        self.add_CDR()  # Psycho
        self.add_NACCMMSE()  # Psycho
        self.add_Memory()  # Psycho
        self.add_MRIInfo_fromCSV()  # MRIInfo

    def merge(self):
        print('\n' + '#' * 10 + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' Merge with MRI files ' + '#' * 10)

        dataAdd = check(self.data)

        dataAdd.insert(loc=0, column='SubID', value='')
        for i in dataAdd.index:
            SubID = dataAdd.loc[i, 'link'].split('.nii')[0]
            assert 'S_{d[RID]}_{d[Image Data ID]}'.format(d=dataAdd.loc[i, :]) == SubID
            dataAdd.loc[i, 'SubID'] = SubID
        assert len(set(dataAdd['SubID'])) == len(dataAdd)

        dataAdd.to_csv(os.path.join(self.Save_path, 'Merged_Data.csv'), index=False)
        print(dataAdd[dataAdd['Label'].isna()].shape[0],
              ' samples (in mri_path_add.csv) have none matched table records.')

    def add_label(self):
        table = pd.read_csv(os.path.join(self.RawTable_path, 'aibl_pdxconv_01-Jun-2018.csv'))
        label_dict = {'NC': [1], 'MCI': [2], 'AD': [3]}

        assert sum([(kk == 'Base' or kk.lower().endswith(' mo')) for kk in self.img['Visit']]) == len(self.img)
        self.img.insert(
            loc=self.img.columns.get_loc('Visit') + 1, column='VISCODE',
            value=self.img['Visit'].apply(
                lambda x: ('m' + x.lower().split(' mo')[0]) if 'mo' in x.lower() else 'bl')
        )
        assert sum([(kk == 'bl' or kk.startswith('m')) for kk in self.img['VISCODE']]) == len(self.img)
        self.img.insert(
            loc=self.img.columns.get_loc('VISCODE') + 1, column='VISMon',
            value=self.img['VISCODE'].apply(
                lambda x: (int(x[1:])) if x.startswith('m') else 0)
        )
        base_cols = ['RID', 'SITEID', 'VISCODE']

        data = pd.merge(self.img, table[base_cols + ['DXCURREN']], how='left', on=['RID', 'VISCODE'])
        data = data.sort_values(by=['RID', 'VISMon']).reset_index(drop=True)
        for i in data.index:
            data.loc[i, 'Label'] = search_value(label_dict, data.loc[i, 'DXCURREN'])
        data.drop(columns=['DXCURREN'], inplace=True)

        return data

    def catch_value(self, table, NACC2ADNI, merge_cols=['RID', 'VISCODE']):
        data = pd.merge(self.data, table[merge_cols + [kk[0] for kk in NACC2ADNI.values()]],
                        how='left', on=merge_cols)
        for nacc_col, (adni_col, convert) in NACC2ADNI.items():
            for i in self.data.index:
                self.data.loc[i, nacc_col] = search_value(convert, data.loc[i, adni_col])

    def add_Demo(self):
        NACC2ADNI = {
            'SEX': ['Sex', {1: ['M'], 2: ['F']}],
            # 1=Male, 2=Female
            'NACCAGE': ['Age', {'same': np.arange(0, 100).astype(int)}],
        }
        for nacc_col, (adni_col, convert) in NACC2ADNI.items():
            for i in self.data.index:
                self.data.loc[i, nacc_col] = search_value(convert, self.data.loc[i, adni_col])
            self.data.drop(columns=[adni_col], inplace=True)

    def add_APOE(self):
        table = pd.read_csv(os.path.join(self.RawTable_path, 'aibl_apoeres_01-Jun-2018.csv'))
        NACC2ADNI = {
            'NACCAPOE': ['APGEN1, APGEN2', {1: [(3, 3)], 2: [(3, 4), (4, 3)], 3: [(3, 2), (2, 3)], 4: [(4, 4)], 5: [(4, 2), (2, 4)], 6: [(2, 2)]}],
            # NACC: 1=e3,e3; 2=e3,e4; 3=e3,e2; 4=e4,e4; 5=e4,e2; 6=e2,e2; 9=Missing/ unknown/ not assessed
            # ADNI: APGEN1, APGEN2 (2, 3, 4)
            'NACCNE4S': ['APGEN1, APGEN2', {0: [(3, 3), (3, 2), (2, 3), (2, 2)], 1: [(3, 4), (4, 3), (4, 2), (2, 4)], 2: [(4, 4)]}],
            # NACC: 0=No e4 allele, 1=1 copy of e4 allele, 2=2 copies of e4 allele, 9=Missing/ unknown/ not assessed"
            # ADNI: APGEN1, APGEN2 (2, 3, 4)
        }
        data = pd.merge(self.data, table[['RID', 'APGEN1', 'APGEN2']],
                        how='left', on=['RID'])
        for nacc_col, (_, convert) in NACC2ADNI.items():
            for i in self.data.index:
                self.data.loc[i, nacc_col] = search_value(convert, (data.loc[i, 'APGEN1'], data.loc[i, 'APGEN2']))

    def add_MedH(self):
        table = pd.read_csv(os.path.join(self.RawTable_path, 'aibl_medhist_01-Jun-2018.csv'))
        NACC2ADNI = {
            'CVHATT': ['MH4CARD', {0: [0], 1: [1]}],
            # NACC: Heart attack/cardiac arrest.
            # 0=Absent; 1=Recent/Active; 2=Remote/Inactive; 9=Unknown; -4=Not available
            # ADNI: Cardiovascular.
            # 1=Yes; 0=No
            'PSYCDIS': ['MHPSYCH', {0: [0], 1: [1]}],
            # NACC: Other psychiatric disorder.
            # ADNI: Psychiatric.
        }
        self.catch_value(table, NACC2ADNI)

    def add_CDR(self):
        table = pd.read_csv(os.path.join(self.RawTable_path, 'aibl_cdr_01-Jun-2018.csv'))
        NACC2ADNI = {
            'CDRGLOB': ['CDGLOBAL', {'same': [0, 0.5, 1, 2, 3]}]
        }
        # NACC: CDRGLOB.
        # 0.0 = No impairment; 0.5 = Questionable impairment; 1.0 = Mild impairment; 2.0 = Moderate impairment; 3.0 = Severe impairment
        # ADNI: CDGLOBAL.
        # 0.0=0.0;0.5=0.5;1.0=1.0;2.0=2.0;3.0=3.0
        self.catch_value(table, NACC2ADNI)

    def add_NACCMMSE(self):
        table = pd.read_csv(os.path.join(self.RawTable_path, 'aibl_mmse_01-Jun-2018.csv'))
        NACC2ADNI = {
            'NACCMMSE': ['MMSCORE', {'same': np.arange(0, 31).astype(int)}],
            # NACC: Total MMSE score (using D-L-R-O-W)
            # 0 - 30
            # 88 = Score not calculated; missing at least one MMSE item
            # 95 = Physical problem
            # 96 = Cognitive/behavior problem
            # 97 = Other problem
            # 98 = Verbal refusal
            # -4 = Not available
            # ADNI: MMSE TOTAL SCORE. 0~30
        }
        self.catch_value(table, NACC2ADNI)

    def add_Memory(self):
        table = pd.read_csv(os.path.join(self.RawTable_path, 'aibl_neurobat_01-Jun-2018.csv'))
        NACC2ADNI = {
            'LOGIMEM': ['LIMMTOTAL', {'same': np.arange(0, 26).astype(int)}],  # 0-25
            'MEMUNITS': ['LDELTOTAL', {'same': np.arange(0, 26).astype(int)}],  # 0-25
        }
        self.catch_value(table, NACC2ADNI)

    def add_MRIInfo_fromCSV(self):
        table_3T = pd.read_csv(os.path.join(self.RawTable_path, 'aibl_mri3meta_01-Jun-2018.csv'))
        table_15T = pd.read_csv(os.path.join(self.RawTable_path, 'aibl_mrimeta_01-Jun-2018.csv'))
        table_3T['MRIFIELD'] = 2
        table_15T['MRIFIELD'] = 1
        merged = pd.concat([table_3T.query('MMCONDCT==1'), table_15T.query('MMCONDCT==1')], axis=0)
        assert len(merged) == len(merged.drop_duplicates(subset=['RID', 'VISCODE']))
        self.data = pd.merge(self.data, merged[['RID', 'SITEID', 'VISCODE', 'EXAMDATE', 'MRIFIELD']], how='left')
        assert sum(['_x' in kk for kk in self.data.columns]) == 0

        max_interval = (pd.to_datetime(self.data['Acq Date']) - pd.to_datetime(self.data['EXAMDATE'])).max().days
        print(f'Interval between EXAMDATE and Acq Date: {max_interval}')
        assert max_interval <= 90


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


def check(merged):
    # Field Strength - MRIFIELD; Manufacturer - MRIMANU; Mfg Model - MRIMODL;
    manu_dict = {
        'GE MEDICAL SYSTEMS': 1,
        'Philips Medical Systems': 2,
        'SIEMENS': 3,
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

        np.nan: np.nan
    }

    for i in merged.index:
        if isNaN(merged.loc[i, 'Field Strength']):
            merged.loc[i, 'MRIFIELDj'] = np.nan
        else:
            field = np.round(float(merged.loc[i, 'Field Strength'].split(' tesla')[0]), 1)
            merged.loc[i, 'MRIFIELDj'] = 1.5 if field == 1.5 else 3.0
            assert ((merged.loc[i, 'MRIFIELDj'] == 1.5 and merged.loc[i, 'MRIFIELD'] == 1)
                    or (merged.loc[i, 'MRIFIELDj'] == 3.0 and merged.loc[i, 'MRIFIELD'] == 2)
                    or isNaN(merged.loc[i, 'MRIFIELD']))

        merged.loc[i, 'MRIMANUj'] = manu_dict[merged.loc[i, 'Manufacturer']]
        # assert merged.loc[i, 'MRIMANUj'] == merged.loc[i, 'MRIMANU']

        merged.loc[i, 'MRIMODLj'] = modl_dict[merged.loc[i, 'Mfg Model']]
        # assert merged.loc[i, 'MRIMODLj'] == merged.loc[i, 'MRIMODL']

        if isNaN(merged.loc[i, 'Slice Thickness']):
            merged.loc[i, 'MRITHICKj'] = np.nan
        else:
            thick = merged.loc[i, 'Slice Thickness']
            merged.loc[i, 'MRITHICKj'] = np.round(float(thick.split(' mm')[0]), 1)

    return merged


if __name__ == '__main__':
    obj = TransformVariables()
    obj.transform()
    obj.merge()
