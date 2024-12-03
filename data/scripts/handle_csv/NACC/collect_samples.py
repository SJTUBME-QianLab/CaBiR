import pandas as pd
import numpy as np
import os
import sys
import shutil
import time
import json
from datetime import date
import nibabel as nib
from joblib import Parallel, delayed
from tqdm import tqdm


class CollectSamples:
    def __init__(self):
        self.NiftiRoot = 'PublicData/NACC/RawData/MRI/'  # unziped files downloaded from NACC
        self.NiftiLink_path = 'MRIProcess/NACC/raw/'  # linked nifti files (to be processed)
        self.RawTable_path = 'PublicData/NACC/CSV/'  # csv files downloaded from NACC

        self.Save_path = os.getcwd().split('scripts')[0] + '/handle_csv/NACC/'  # handled csvs will be saved here
        os.makedirs(self.Save_path, exist_ok=True)
        self.save_log = True
        self.backup()

    def backup(self):
        timestamp = time.strftime('%Y%m%d%H%M%S')
        os.makedirs(os.path.join(os.getcwd(), 'backup'), exist_ok=True)
        shutil.copy2('collect_samples.py', os.path.join(os.getcwd(), 'backup', f"collect_samples_{timestamp}.py"))
        if self.save_log:
            sys.stdout = Logger("collect_samples", timestamp)

    def main(self):
        if not os.path.isfile(os.path.join(self.Save_path, 'Table_MRI.csv')):
            # Read raw table; Add diagnosis: NC/MCI/AD/None
            RawTable = self.AddLabel()
            # Add MRI data
            self.MatchMRI(RawTable)
        TableMRI = pd.read_csv(os.path.join(self.Save_path, 'Table_MRI.csv'))
        print('TableMRI: ', len(TableMRI))  # 8104

        if not os.path.isfile(os.path.join(self.Save_path, 'Table_MRI_File.csv')):
            self.SearchMRIT1(TableMRI)
        TableMRIFile = pd.read_csv(os.path.join(self.Save_path, 'Table_MRI_File.csv'))

        print(TableMRIFile.shape, 'File number: ', sum(TableMRIFile['FileNum']))
        print('Sample & T1 nifti files:\t0\t1\t>1\ttotal')
        print('sample: \t{:d}\t{:d}\t{:d}\t{:d}'.format(
            len(TableMRIFile.query('FileNum==0')), len(TableMRIFile.query('FileNum==1')),
            len(TableMRIFile.query('FileNum>1')), len(TableMRIFile))
        )
        print('T1 nifti: \t{:d}\t{:d}\t{:d}\t{:d}'.format(
            sum(TableMRIFile.query('FileNum==0')['FileNum']), sum(TableMRIFile.query('FileNum==1')['FileNum']),
            sum(TableMRIFile.query('FileNum>1')['FileNum']), sum(TableMRIFile['FileNum']))
        )

    def AddLabel(self):
        Table = pd.read_csv(os.path.join(self.RawTable_path, 'investigator_nacc63.csv'))
        # Select necessary columns
        NACCCols = ['NACCID', 'NACCVNUM', 'VISITYR', 'VISITMO', 'VISITDAY', 'NACCUDSD', 'NACCALZD']
        Table = Table[NACCCols]

        Table['Label'] = 'None'
        for i in range(len(Table)):
            alz = Table.loc[i, 'NACCALZD']  # alzheimer diagnosis
            # 0 = No; 1 = Yes; 8 = No cognitive impairment
            dementia = Table.loc[i, 'NACCUDSD']  # cognitive status
            # 1 = Normal cognition; 2 = Impaired-not-MCI; 3 = MCI; 4 = Dementia

            if alz == 8 and dementia == 1:
                Table.loc[i, 'Label'] = 'NC'
            elif alz == 1 and dementia == 3:
                Table.loc[i, 'Label'] = 'MCI'
            elif alz == 1 and dementia == 4:
                Table.loc[i, 'Label'] = 'AD'

        return Table

    def MatchMRI(self, Table):
        visDate = ['VISITYR', 'VISITMO', 'VISITDAY']
        Raw2Row = Table.groupby('NACCID').groups

        mriDate = ['MRIYR', 'MRIMO', 'MRIDY']
        dataMRI = pd.read_csv(os.path.join(self.RawTable_path, 'investigator_mri_nacc63.csv'))
        dataMRI = dataMRI[['NACCID', 'NACCVNUM', 'NACCMNUM', 'NACCMRDY', 'NACCMRFI', 'MRIT1', 'NACCNIFT', 'NACCMVOL']
                          + mriDate]

        MRI2Row = dataMRI.groupby('NACCID').groups
        TableMRI = []
        for ID, row in MRI2Row.items():
            if ID not in Raw2Row.keys():
                continue
            diagTimes = Table.loc[Raw2Row[ID], visDate]
            for ii in row:
                mriTime = dataMRI.loc[ii, mriDate]
                idx, diff = find_closest(mriTime, diagTimes)
                if idx is None:
                    continue
                else:
                    idx = Raw2Row[ID][idx]
                if not (Table.loc[idx, 'NACCVNUM'] == dataMRI.loc[ii, 'NACCVNUM']):
                    assert dataMRI.loc[ii, 'NACCMRDY'] == diff
                    print('NACCVNUM from Table and MRI do not match!')
                    print(ID, mriTime.values, dataMRI.loc[ii, 'NACCVNUM'], Table.loc[idx, 'NACCVNUM'],
                          'Cal:', diff, 'NACCMRDY:', dataMRI.loc[ii, 'NACCMRDY'])
                TableMRI.append(pd.merge(dataMRI.loc[[ii], :].drop(['NACCVNUM'], axis=1),
                                         Table.loc[[idx], :]))
        TableMRI = pd.concat(TableMRI, axis=0)

        # remove samples with Label='None'
        TableMRI = TableMRI.query("Label!='None'").reset_index(drop=True)
        # save TableMRI
        print(len(TableMRI))  # 8104
        TableMRI.to_csv(os.path.join(self.Save_path, 'Table_MRI.csv'), index=False)

    def MatchCSF(self, Table):
        visDate = ['VISITYR', 'VISITMO', 'VISITDAY']
        Raw2Row = Table.groupby('NACCID').groups

        csfDate = ['CSFLPYR', 'CSFLPMO', 'CSFLPDY']
        dataCSF = pd.read_csv(os.path.join(self.RawTable_path, 'investigator_fcsf_nacc63.csv'))
        dataCSF = dataCSF[['NACCID'] + csfDate]
        CSF2Row = dataCSF.groupby('NACCID').groups

        TableCSF = []
        for ID, row in CSF2Row.items():
            if ID not in Raw2Row.keys():
                continue
            diagTimes = Table.loc[Raw2Row[ID], visDate]
            for ii in row:
                csfTime = dataCSF.loc[ii, csfDate]
                idx, diff = find_closest(csfTime, diagTimes)
                if idx is None:
                    continue
                else:
                    idx = Raw2Row[ID][idx]
                TableCSF.append(pd.merge(dataCSF.loc[[ii], :], Table.loc[[idx], :]))
        TableCSF = pd.concat(TableCSF, axis=0)
        # save TableMRI
        print(len(TableCSF))
        TableCSF.to_csv(os.path.join(self.Save_path, 'Table_MRI_File_CSF.csv'), index=False)

    def SearchMRIT1(self, TableMRI):
        # T1 nifti file exists
        df = TableMRI.query('MRIT1==1 and NACCNIFT==1').reset_index(drop=True)
        print('Shape of Table_MRI_File.csv: ',  df.shape)  # (6559, 17)

        def sub(i):
            pathParent = os.path.join(self.NiftiRoot, 'within1yr', 'nifti', df.loc[i, 'NACCMRFI'].split('.zip')[0])
            if not os.path.isdir(pathParent):
                return None
            T1_list = SearchFile(pathParent)
            sub_RawPath = []
            link_list = []
            for k, pathi in enumerate(T1_list):
                assert os.path.isfile(pathi)
                link = '.'.join(
                    [str(kk) for kk in df.loc[i, ['NACCID', 'NACCVNUM', 'NACCMNUM']]] + [str(k + 1)]) + '.nii'
                path = pathi.split(self.NiftiRoot)[1]
                sub_RawPath.append(list(df.loc[i, :].values) + [link, path.replace('.json', '.nii')])
                link_list.append(link)
            if len(link_list) == 0:
                print('-' * 10 + ' FIND NO T1', df.loc[i, 'NACCMRFI'])
                return None
            else:
                return i, link_list, sub_RawPath

        # further filter
        packages = Parallel(n_jobs=10)(delayed(sub)(i) for i in tqdm(range(len(df))))
        df['SoftLink'] = ''
        df['FileNum'] = 0
        RawPath = []
        for pp in packages:
            if pp is not None:
                df.loc[pp[0], 'SoftLink'] = '; '.join(pp[1])
                df.loc[pp[0], 'FileNum'] = len(pp[1])
                RawPath.extend(pp[2])
        RawPath = pd.DataFrame(RawPath, columns=list(df.columns)[:-2] + ['link', 'path'])

        # save
        df.to_csv(os.path.join(self.Save_path, 'Table_MRI_File.csv'), index=False)
        RawPath.to_csv(os.path.join(self.Save_path, 'mri_path.csv'), index=False)
        print('File number: ', len(RawPath))  # 7610

    def CreateLink(self):
        MRIPath = pd.read_csv(os.path.join(self.Save_path, 'mri_path.csv'))

        def sub(i):
            file1 = os.path.join(self.NiftiRoot, MRIPath.loc[i, 'path'].replace('/within1yr', 'within1yr'))
            file2 = os.path.join(self.NiftiLink_path, MRIPath.loc[i, 'link'])
            assert os.path.isfile(file1)
            assert os.path.isfile(file2)

            file1 = file1.replace('.nii', '.json')
            file2 = file2.replace('.nii', '.json')
            assert os.path.isfile(file1)
            assert os.path.isfile(file2)

        Parallel(n_jobs=10, require='sharedmem')(delayed(sub)(i) for i in tqdm(range(len(MRIPath))))

    def AddJson(self):
        MRIPath = pd.read_csv(os.path.join(self.Save_path, 'mri_path.csv'))
        MRIPath['MRIFIELDj'] = 0
        MRIPath['MRIMANUj'] = ''
        MRIPath['MRIMODLj'] = ''
        MRIPath['MRITHICKj'] = 0
        MRIPath['SeriesDescription'] = ''

        def sub(df0):
            dfi = df0.copy()
            pathi = os.path.join(self.NiftiLink_path, dfi['link'].replace('.nii', '.json'))
            assert os.path.isfile(pathi)
            jj = json.load(open(pathi, 'r'))

            try:
                field = jj['MagneticFieldStrength']
                if field == 15000:
                    field = 1.5
                dfi['MRIFIELDj'] = field
            except KeyError:
                pass
            try:
                dfi['MRIMANUj'] = jj['Manufacturer']
            except KeyError:
                pass
            try:
                dfi['MRIMODLj'] = jj['ManufacturersModelName']
            except KeyError:
                pass
            try:
                dfi['MRITHICKj'] = jj['SliceThickness']
            except KeyError:
                pass
            try:
                dfi['SeriesDescription'] = jj['SeriesDescription']
            except KeyError:
                pass
            return dfi
        df = Parallel(n_jobs=10, require='sharedmem')(delayed(sub)(MRIPath.loc[i, :])
                                                      for i in tqdm(range(len(MRIPath))))
        MRIPath = pd.concat(df, axis=1).T.sort_index(axis=0)

        mriTable = pd.read_csv(os.path.join(self.RawTable_path, 'investigator_mri_nacc63.csv'))
        df_merge = pd.merge(MRIPath, mriTable[['NACCID', 'NACCMRFI', 'MRIFIELD', 'MRIMANU', 'MRIMODL']], how='inner')
        pd.testing.assert_frame_equal(df_merge.iloc[:, :-3], MRIPath)
        df_merge = df_merge.sort_values(['NACCID', 'NACCMRFI']).reset_index(drop=True)
        df = modifyTableJson(df_merge)

        df.to_csv(os.path.join(self.Save_path, 'mri_path_add.csv'), index=False)


def find_closest(mri, diags):
    min_diff = 1000000
    ans = None
    mriDate = date(*mri)
    for i in range(len(diags)):
        diagDate = date(*diags.iloc[i, :])
        diff = int((mriDate - diagDate).days)
        if abs(diff) < min_diff and abs(diff) <= 180:  # within +/- 6 months
            min_diff = diff
            ans = i
    return ans, min_diff


def T1Filter(FileName):
    T1_keys = ['MPRAGE', 'MPrage', 'MP-RAGE', 'MPR', 'SPGR', 'spgr', 'T1', 't1']
    # MP-RAGE: magnetization prepared - rapid acquisition gradient echo
    # SPGR: Spoiled Gradient-Recalled Echo
    return sum([(kk in FileName) for kk in T1_keys]) > 0


def catchShape(pathi):
    pathi = pathi.replace('.json', '.nii')
    image_obj = nib.load(pathi.replace('.json', '.nii'))
    shape = image_obj.get_fdata().shape
    if len(shape) < 3 or min(shape) < 60:
        return 'SHAPE'
    else:
        return 'SUCCESS' + str(shape)


def checkT1(pathi):
    if not pathi.endswith('.json'):
        return 'JSON'
    if not (os.path.isfile(pathi) and os.path.isfile(pathi.replace('.json', '.nii'))):
        return 'JSON'
    try:
        with open(pathi, 'r') as file:
            jj = json.load(file)
    except:
        return 'JSON'

    if 'SeriesDescription' not in jj.keys():
        if jj['MRAcquisitionType'] == '3D' and jj['SliceThickness'] < 2:
            print('Miss SeriesDescription.\t' + pathi)
            return catchShape(pathi)
        else:
            return 'FAIL'
    elif T1Filter(jj['SeriesDescription']):
        if 'MRAcquisitionType' in jj.keys() and 'SliceThickness' in jj.keys():
            if jj['SliceThickness'] < 2 and jj['MRAcquisitionType'] == '3D':
                return catchShape(pathi)
            else:
                return 'FAIL'
        elif 'MRAcquisitionType' not in jj.keys() and 'SliceThickness' in jj.keys():
            if jj['SliceThickness'] < 2:
                print('Miss MRAcquisitionType.\t' + pathi)
                return catchShape(pathi)
            else:
                return 'Thick' + str(jj['SliceThickness'])
        elif 'MRAcquisitionType' in jj.keys() and 'SliceThickness' not in jj.keys():
            if jj['MRAcquisitionType'] == '3D':
                print('Miss SliceThickness.\t' + pathi)
                return catchShape(pathi)
            else:
                return 'Type' + jj['MRAcquisitionType']
        else:
            return 'FAIL'
    else:
        return 'FAIL'


def modifyTableJson(df_merge):
    df = df_merge.copy()

    # MRIFIELD --------------------------------------------------------------------------------
    # 1 = 1.5; 2 = 3.0; 5 = Other;
    # 7 = Field strength varies across images; 8 = Not applicable / no MRI available; 9 = Missing / unknown;
    for i in range(len(df)):
        mj = df.loc[i, 'MRIFIELDj']
        mt = df.loc[i, 'MRIFIELD']
        assert not (isNaN(mj) or isNaN(mt))
        if mt in [7, 9]:
            if 1.49 <= mj <= 1.5:
                mj = 1.5
            elif mj == 3:
                pass
            else:
                print(mj, mt)
        else:
            if 1.49 <= mj <= 1.5:
                mj = 1.5
                assert mt == 1
            elif mj == 3:
                assert mt == 2
            else:
                print(mj, mt)
                assert mt == 5
        df.loc[i, 'MRIFIELDj'] = mj

    # MRIMANU ---------------------------------------------------------------------------------
    # 1 = GE; 2 = Siemens; 3 = Phillips;
    # 5 = Other; 8 = Not applicable / no MRI available; 9 = Missing / unknown;
    for i in range(len(df)):
        mj = df.loc[i, 'MRIMANUj']
        mt = df.loc[i, 'MRIMANU']
        if mt == 9:
            if mj == 'GE':
                mt = 1
            elif mj == 'Siemens':
                mt = 2
            elif mj == 'Phillips' or mj == 'Philips':
                mt = 3
            elif mj == '':
                print(i, df.loc[i, 'link'], mj, mt)
            else:
                print(i, df.loc[i, 'link'], mj, mt)
                raise ValueError
        else:
            if mj == 'GE':
                try:
                    assert mt == 1
                except AssertionError:
                    print(i, df.loc[i, 'link'], mj, mt)
                    mt = 1
            elif mj == 'Siemens':
                try:
                    assert mt == 2
                except AssertionError:
                    print(i, df.loc[i, 'link'], mj, mt)
                    mt = 2
            elif mj == 'Phillips' or mj == 'Philips':
                try:
                    assert mt == 3
                except AssertionError:
                    print(i, df.loc[i, 'link'], mj, mt)
                    mt = 3
            elif mj == '':
                pass
            else:
                print(i, df.loc[i, 'link'], mj, mt)
                raise ValueError
        df.loc[i, 'MRIMANUj'] = mt

    # MRIMODL ---------------------------------------------------------------------------------
    # 1 = DISCOVERY MR 750; 2 = GENESIS SIGNA; 3 = SIGNA HDxt; 4 = Trio Tim; 5 = Eclipse 1.5T;
    # 6 = Allegra; 7 = SIGNA EXCITE; 8 = SIGNA; 9 = GEMINI; 10 = Ingenuity; 11 = Sonata; 12 = Skyra; 13 = Signa HDx;
    # 14 = Achieva; 15 = Prisma; 16 = Verio; 88 = Not applicable / no MRI available; 99 = Missing / unknown
    custom = {
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
    }
    for i in range(len(df)):
        mj = df.loc[i, 'MRIMODLj']
        mt = df.loc[i, 'MRIMODL']
        if mt == 99:
            if mj in custom:
                mt = custom[mj]
            elif mj == 'DicomCleaner' or mj == '':
                pass
            else:
                print(df.loc[i, 'link'], mj, mt)
                raise ValueError
        else:
            if 'signa' in mj.lower():
                try:
                    assert mt in [2, 3, 7, 8, 13]
                except AssertionError:
                    print(df.loc[i, 'link'], mj, mt)
                mt = custom[mj]
            elif mj in custom:
                try:
                    assert mt == custom[mj]
                except AssertionError:
                    print(df.loc[i, 'link'], mj, mt)
                mt = custom[mj]
            elif mj == 'DicomCleaner' or mj == '':
                print(df.loc[i, 'link'], mj, mt)
            else:
                print(df.loc[i, 'link'], mj, mt)
                raise ValueError
        df.loc[i, 'MRIMODLj'] = mt

    for i in set(custom.values()):
        models = set(df_merge[df['MRIMODLj'] == i]['MRIMODLj'])
        for mm in models:
            assert mm == 'DicomCleaner' or custom[mm] == i
    models = set(df_merge[df['MRIMODLj'] == 99]['MRIMODLj'])
    assert models - {np.nan, 'DicomCleaner', ''} == set()

    return df


def SearchFile(curPath):
    fileList = []
    names = sorted(os.listdir(curPath))
    for name in names:
        path = os.path.join(curPath, name)
        if os.path.isdir(path):
            innerFileList = SearchFile(path)
            fileList += innerFileList
        else:
            assert os.path.isfile(path)
            message = checkT1(path)
            if message.startswith('SUCCESS'):
                shape = [int(kk) for kk in message.split('(')[1].split(')')[0].split(', ')]
                if len(shape) > 3:
                    print(message, path)
                fileList.append(path)
            elif message.startswith('Thick') or message.startswith('Type'):
                print(message, path)
    return fileList


class Logger:
    def __init__(self, filename, timestamp):
        self.terminal = sys.stdout
        files = {kk: int(kk.split('_')[-1].split('.log')[0]) for kk in os.listdir(os.getcwd())
                 if kk.endswith('.log') and kk.startswith(filename)}
        files = sorted(files.items(), key=lambda x: x[1])
        new_name = f"{filename}_{timestamp}.log"
        if len(files) >= 1:
            file = files[-1][0]
            shutil.copy2(file, new_name)
            self.log = open(new_name, "a")
            self.log.write('\n\n')
            # print(files)
            self.write(f"New file: {new_name}\nCopy from old file: {file}\n")
            for ff in files:
                os.remove(ff[0])
        else:
            self.log = open(new_name, "a")
            self.write(f"First file: {new_name}\n")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def print_format(series):
    string = '; '.join([f"{kk}: {series[kk]}" for kk in series.index])
    return string


def isNaN(value):
    if isinstance(value, float):
        return np.isnan(value)
    elif isinstance(value, str):
        return value.lower() == 'nan' or value.lower() == 'none' or value == ''
    else:
        return False


if __name__ == '__main__':
    obj = CollectSamples()
    obj.main()
    obj.CreateLink()
    obj.AddJson()
