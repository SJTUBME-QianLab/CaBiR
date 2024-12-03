import pandas as pd
import numpy as np
import os
import sys
import shutil
import time
import json
import nibabel as nib
from joblib import Parallel, delayed
from tqdm import tqdm
import re


class CollectSamples:
    def __init__(self):
        self.NiftiLink_path = 'MRIProcess/OASIS3/raw/'  # linked nifti files (to be processed)
        self.NiftiRoot = 'PublicData/OASIS/OASIS3/'  # dcm files downloaded from OASIS3
        self.RawTable_path = 'PublicData/OASIS/files/OASIS3_data_files/scans/'  # csv files downloaded from OASIS3

        self.Save_path = os.getcwd().split('scripts')[0] + '/handle_csv/OASIS3/'  # handled csvs will be saved here
        os.makedirs(self.Save_path, exist_ok=True)
        self.save_log = True
        self.backup()

    def backup(self):
        timestamp = time.strftime('%Y%m%d%H%M%S')
        os.makedirs(os.path.join(os.getcwd(), 'backup'), exist_ok=True)
        shutil.copy2('collect_samples.py', os.path.join(os.getcwd(), 'backup', f"collect_samples_{timestamp}.py"))
        if self.save_log:
            sys.stdout = Logger("collect_samples", timestamp)

    def SearchMRIT1(self):
        print('\n' + '#' * 10 + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' Search MRI T1w ' + '#' * 10)

        mri_path = []
        for sub_day in sorted(os.listdir(self.NiftiRoot)):
            path_i = os.path.join(self.NiftiRoot, sub_day)
            assert os.path.isdir(path_i)
            k = 1
            run = 0
            for anat in sorted(os.listdir(path_i)):
                path_i = os.path.join(self.NiftiRoot, sub_day, anat)
                assert os.path.isdir(path_i)
                for file in os.listdir(os.path.join(self.NiftiRoot, sub_day, anat)):
                    if file.endswith('.json'):
                        assert os.path.isfile(os.path.join(path_i, file.replace('.json', '.nii.gz')))
                        sub, day = sub_day.split('_MR_')
                        assert sub in file and day in file
                        mri_path.append({
                            'subject_id': sub,
                            'label': sub_day,
                            'filename': file,
                            'link': f'{sub}.{day}.{k}.nii.gz',
                            'path': os.path.join(sub_day, anat, file.replace('.json', '.nii.gz'))
                        })
                        k += 1
                        if 'run' in file:
                            assert run < int(file.split('_')[-2].split('run-')[1].strip())
        mri_path = pd.DataFrame(mri_path)
        pd.testing.assert_frame_equal(mri_path, mri_path.sort_values(by=['link']).reset_index(drop=True))
        # # pd.testing.assert_frame_equal(mri_path, mri_path.sort_values(by=['label', 'filename']))
        # ind = np.where(mri_path.index != mri_path.sort_values(by=['label', 'filename']).index)
        # print(mri_path.iloc[ind][['link', 'label', 'filename']])
        # print(mri_path.sort_values(by=['label', 'filename']).iloc[ind][['link', 'label', 'filename']])
        mri_path.to_csv(os.path.join(self.Save_path, 'mri_path.csv'), index=False)

    def CreateLink(self):
        print('\n' + '#' * 10 + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' Create Link ' + '#' * 10)
        MRIPath = pd.read_csv(os.path.join(self.Save_path, 'mri_path.csv'))
        os.makedirs(self.NiftiLink_path, exist_ok=True)

        def sub(i):
            file1 = os.path.join(self.NiftiRoot, MRIPath.loc[i, 'path'])
            file2 = os.path.join(self.NiftiLink_path, MRIPath.loc[i, 'link'])
            assert os.path.isfile(file1)
            # assert not os.path.isfile(file2)
            if not os.path.isfile(file2):
                cmd = f'ln \"{file1}\" \"{file2}\"'
                # print(cmd)
                os.system(cmd)

        Parallel(n_jobs=10, require='sharedmem')(delayed(sub)(i) for i in tqdm(range(len(MRIPath))))

    def AddShape(self):
        print('\n' + '#' * 10 + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' Add Shape ' + '#' * 10)
        MRIPath = pd.read_csv(os.path.join(self.Save_path, 'mri_path.csv'))
        MRIPath['shape'] = ''
        MRIPath[['resolX', 'resolY', 'resolZ']] = np.nan

        def sub(i):
            path = os.path.join(self.NiftiRoot, MRIPath.loc[i, 'path'])
            size, resol = catchShape(path)
            MRIPath.loc[i, 'shape'] = size
            MRIPath.loc[i, ['resolX', 'resolY', 'resolZ']] = resol

        Parallel(n_jobs=10, require='sharedmem')(delayed(sub)(i) for i in tqdm(range(len(MRIPath))))

        MRIPath.to_csv(os.path.join(self.Save_path, 'mri_path_add.csv'), index=False)

    def AddJson(self):
        print('\n' + '#' * 10 + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' Add Json from csv' + '#' * 10)

        MRIPath0 = pd.read_csv(os.path.join(self.Save_path, 'mri_path_add.csv'))
        mri_file = pd.read_csv(
            os.path.join(self.RawTable_path, 'MRI-json-MRI_json_information/resources/csv/files/OASIS3_MR_json.csv')
        )
        MRIPath = pd.merge(MRIPath0, mri_file, on=['subject_id', 'label', 'filename'], how='inner')
        print(f'Number of nifti files: {len(MRIPath0)}, After merged with json.csv: {len(MRIPath)}')
        # assert len(merged) == len(mri_path)  4115
        assert MRIPath['scan category'].unique() == ['T1w']
        MRIPath.dropna(how='all', axis=1, inplace=True)  # 去掉全是nan的列

        def sub(i):
            row = MRIPath.loc[i]
            json_info = json.load(open(os.path.join(self.NiftiRoot, row['path'].replace('.nii.gz', '.json')), 'r'))
            for k, v in json_info.items():
                assert k in MRIPath.columns
                if k == 'AcquisitionTime':
                    continue
                if not check_json(k, row[k], v):
                    print(f"Conflict: {row['filename']} {k} ......\t {row[k]} != {v}")
            for k in MRIPath.columns:
                if k in MRIPath0.columns or k in ['acccession', 'release version', 'scan category']:
                    continue
                if k not in json_info and not isNaN(MRIPath.loc[i, k]):
                    print(f"Missing: {row['filename']} {k} ......\t {row[k]}")

        Parallel(n_jobs=10, require='sharedmem')(delayed(sub)(i) for i in tqdm(range(len(MRIPath))))
        assert [kk in MRIPath.columns for kk in ['MagneticFieldStrength', 'Manufacturer', 'ManufacturersModelName', 'SliceThickness']] == [True] * 4

        MRIPath.to_csv(os.path.join(self.Save_path, 'mri_path_add.csv'), index=False)


def catchShape(path):
    image_obj = nib.load(path)
    shape = image_obj.get_fdata().shape
    size = image_obj.header['dim'][1:4]
    assert (shape == size).all(), f'WRONG: {shape} != {size}'
    resol = image_obj.header['pixdim'][1:4]
    if len(size) < 3 or min(size) < 60 or max(resol) > 2:
        print(f"{os.path.basename(path)} NOT satisfied, size={size}, resol={resol}")
    
    # thick = catchThick(size, resol)
    size = ', '.join('{:d}'.format(kk) for kk in size)
    # resol = ', '.join(kk for kk in resol)
    return size, resol


def check_json(k, v1, v2):
    if isNaN(v1):
        return False
    try:
        v2 = float(v2)
    except:
        pass
    if isinstance(v2, str):
        return str(v1) == v2
    if isinstance(v2, float):
        return np.isclose(v1, v2)
    if isinstance(v2, int):
        return int(v1) == v2
    if isinstance(v2, list):
        return v1 == str(v2)
    return False


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
    obj.SearchMRIT1()
    obj.CreateLink()
    obj.AddShape()
    obj.AddJson()
