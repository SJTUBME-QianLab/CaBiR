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
import requests
from fake_useragent import UserAgent  # conda install -c conda-forge fake-useragent


class CollectSamples:
    def __init__(self):
        self.NiftiRoot = 'PublicData/AIBL/raw_nifti/'  # converted nifti files from dcm files
        self.NiftiLink_path = 'MRIProcess/AIBL/raw/'  # linked nifti files (to be processed)
        self.dcmRoot = 'PublicData/AIBL/AIBL/'  # dcm files downloaded from AIBL

        self.Save_path = os.getcwd().split('scripts')[0] + '/handle_csv/AIBL/'  # handled csvs will be saved here
        os.makedirs(self.Save_path, exist_ok=True)
        self.save_log = True
        self.backup()

    def backup(self):
        timestamp = time.strftime('%Y%m%d%H%M%S')
        os.makedirs(os.path.join(os.getcwd(), 'backup'), exist_ok=True)
        shutil.copy2('collect_samples.py', os.path.join(os.getcwd(), 'backup', f"collect_samples_{timestamp}.py"))
        if self.save_log:
            sys.stdout = Logger("collect_samples", timestamp)

    def dcm2nii(self):
        print('\n' + '#' * 10 + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' dcm2nii ' + '#' * 10)

        file_list = pd.read_csv(os.path.join(self.dcmRoot, '..', 'AIBL_3DT1_12_06_2023.csv'))
        file_list.drop(['Downloaded'], axis=1, inplace=True)

        def sub(i):
            row = file_list.iloc[i].copy()
            imgID = row['Image Data ID']
            subID = row['Subject']
            package = f'S_{subID}_{imgID}'
            description = row['Description'].replace(' ', '_')
            mm, dd, yy = [int(kk) for kk in row['Acq Date'].split('/')]
            time_str = f'{yy:04d}-{mm:02d}-{dd:02d}'
            try:
                dcm_dir0 = os.path.join(self.dcmRoot, str(subID), description)
                dcm_dir = ''
                for pp in os.listdir(dcm_dir0):
                    if pp.startswith(time_str):
                        dcm_dir = os.path.join(dcm_dir0, pp, imgID)
                assert os.path.isdir(dcm_dir), f'WRONG: {dcm_dir} is not dir'
            except FileNotFoundError as e:
                return str(e)
            except AssertionError as e:
                return str(e)

            out_dir = os.path.join(self.NiftiRoot, package)
            os.makedirs(out_dir, exist_ok=True)
            if len(os.listdir(out_dir)) == 0:
                cmd = f'dcm2nii -x N -r N -d Y -e N -i N -p N -f Y -o {out_dir} {dcm_dir}'  # >> ./dcm2nii.txt
                status = os.system(cmd)
                if status != 0:
                    return f'WRONG: {cmd}'

            # confirm
            assert len(os.listdir(out_dir)) == 1, f'WRONG: {out_dir} != 1 file'
            nii_file = os.listdir(out_dir)[0]
            assert nii_file.startswith(f'{yy:04d}{mm:02d}{dd:02d}'), f'WRONG date: {nii_file[:8]}'
            key_str = f'AIBL_{subID}_MR_{description}'
            assert nii_file[15:].startswith(key_str), f'WRONG: {nii_file} != {key_str}'
            assert nii_file.endswith(f'{imgID}.nii.gz'), f'WRONG: {nii_file} != {imgID}.nii.gz'
            row['path'] = os.path.join(package, nii_file)
            row['link'] = f'{package}.nii.gz'
            row['series'] = int(nii_file.split('_')[-2].split('S')[1])
            return row

        res = Parallel(n_jobs=10)(delayed(sub)(i) for i in tqdm(range(len(file_list))))
        for r in res:
            if isinstance(r, str):
                print(r)
        success = pd.concat([kk for kk in res if not isinstance(kk, str)], axis=1).T
        print(success.columns)
        print(file_list.shape, ' -> ', success.shape)  # (1300, 11)  ->  (1297, 14) remove 3 phantom
        success['Date'] = pd.to_datetime(success['Acq Date'])
        success = success.sort_values(by=['Subject', 'Date']).drop(columns=['Date'], axis=1).reset_index(drop=True)
        success.to_csv(os.path.join(self.Save_path, 'mri_path.csv'), index=False)
        # Parallel(n_jobs=10)(delayed(sub)(i) for i in tqdm(range(5)))

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
        # Extract the shape and pixel size of the 3D image from the NII file (for extracting layer thickness), and
        # add it to mri_path.csv
        MRIPath['shape'] = ''
        MRIPath[['resolX', 'resolY', 'resolZ']] = np.nan

        def sub(i):
            path = os.path.join(self.NiftiRoot, MRIPath.loc[i, 'path'])
            size, resol = catchShape(path)
            MRIPath.loc[i, 'shape'] = size
            MRIPath.loc[i, ['resolX', 'resolY', 'resolZ']] = resol

        Parallel(n_jobs=10, require='sharedmem')(delayed(sub)(i) for i in tqdm(range(len(MRIPath))))

        MRIPath.to_csv(os.path.join(self.Save_path, 'mri_path_add.csv'), index=False)

    def AddXML(self):
        print('\n' + '#' * 10 + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' Add XML ' + '#' * 10)
        MRIPath = pd.read_csv(os.path.join(self.Save_path, 'mri_path_add.csv'))

        def sub(i):
            row = MRIPath.loc[i]
            imageID = row['Image Data ID']
            seriesID = MRIPath.loc[i, 'series']
            if 'protocol' in MRIPath.columns and str(MRIPath.loc[i, 'protocol']) != 'nan':
                dfi = MRIPath.iloc[[i], (MRIPath.columns.get_loc('resolZ') + 1):]
                dfi['Image Data ID'] = imageID
                return dfi
            text = get_protocol(seriesID, remind=False)
            items = handleStr(text, row)
            if items is None:
                print(i, imageID, 'fail')
                return None
            items['Image Data ID'] = imageID
            dfi = pd.DataFrame(items, index=[0])
            print(i, imageID, 'success')
            return dfi

        save = []
        step = 20
        for start in range(0, len(MRIPath), step):
            res = Parallel(n_jobs=10, require='sharedmem')(delayed(sub)(i) for i in
                                                           tqdm(range(start, min(start+step, len(MRIPath)))))
            save.extend([kk for kk in res if kk is not None])
            MRIPath_add = pd.merge(MRIPath.iloc[:, :(MRIPath.columns.get_loc('resolZ') + 1)],
                                   pd.concat(save, axis=0), on='Image Data ID', how='left')
            MRIPath_add.to_csv(os.path.join(self.Save_path, 'mri_path_add.csv'), index=False)


def find_dir(path, name, imgID):
    assert os.path.isdir(path), f'WRONG: {path} is not dir'
    for file in os.listdir(path):
        if file.startswith(name):
            find = os.path.join(path, file)
            assert os.path.isdir(find), f'WRONG: {find} is not dir'
            sub_package = os.listdir(find)
            for img in sub_package:
                if img != imgID:
                    continue
                find = os.path.join(find, img)
                assert os.path.isdir(find), f'WRONG: {find} is not dir'
                nii = os.listdir(find)
                assert len(nii) == 1, f'WRONG: {nii} >= 1'
                assert nii[0].endswith('.nii'), f'WRONG: {nii[0]} != \*.nii'
                return os.path.join(find, nii[0])
    return None


def catchShape(path):
    image_obj = nib.load(path)
    shape = image_obj.get_fdata().shape
    size = image_obj.header['dim'][1:4]
    assert (shape == size).all(), f'WRONG: {shape} != {size}'
    resol = image_obj.header['pixdim'][1:4]
    if len(size) < 3 or min(size) < 60:
        print(f"{os.path.basename(path)} NOT satisfied, size={size}, resol={resol}")
        return '', ''
    else:
        size = ', '.join('{:d}'.format(kk) for kk in size)
        return size, resol


def get_protocol(seriesID, remind=False):
    # ！！！！！！！！ 不开VPN ！！！！！！！！！！！！！！！！！
    ua = UserAgent()
    request_url = 'https://ida.loni.usc.edu/pages/access/imageDetail/imageDetails.jsp?project=AIBL&seriesId=%d' % seriesID
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-encoding': 'gzip, deflate, br',
        'Accept-language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        "User-Agent": ua.random
        #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
    }
    cookie_dict = dict(
        JSESSIONID='',
        _ga_HWP2E80XXT='',
    )
    cookies = requests.utils.cookiejar_from_dict(cookie_dict, cookiejar=None, overwrite=True)
    retries = 0
    while retries < 20:
        try:
            response = requests.get(url=request_url, headers=headers, cookies=cookies)
            response.raise_for_status()
            assert response.text != ''
            assert 'IDA account' not in response.text
            return response.text
        except AssertionError or requests.exceptions.RequestException:
            retries += 1
            if remind:
                print(f"Request failed, retrying for the {retries} time(s)...")
    return response.text


def handleStr(text, row):
    if text == '':
        return None
    check = re.findall('Subject ID:([0-9\s]+)\n\s+</td>', text)
    if len(check) != 1 or row['Subject'] != int(check[0].strip()):
        print('NOT Find Subject ID: ', row['Subject'])
        return None
    sex_dict = {'Female': 'F', 'Male': 'M', 'Unknown': 'X'}
    check = re.findall('<td><strong>Sex</strong></td>\n\s+<td>:</td>\n\s+<td>([A-Za-z]+)</td>\n', text)
    if len(check) != 1 or row['Sex'] != sex_dict[check[0]]:
        print('NOT Find Sex: ', row['Sex'])
        return None
    check = re.findall('<td><strong>Visit</strong></td>\n\s+<td>:</td>\n\s+<td>(.*)</td>\n', text)
    check = re.findall('[A-Za-z0-9\s]+\(([A-Z0-9\s]+)\)', check[0])
    if len(check) != 1 or row['Visit'].upper() != check[0]:
        print('NOT Find Visit: ', row['Visit'])
        return None

    body = re.findall('Imaging Protocol\n\s+</span>\n\s+</td>\n\s+<td valign="top">:</td>\n\s+<td>\n\s+(.*)\n\s+</td>', text)
    if len(body) != 1:
        print(text)
        return None
    body = body[0]
    items = dict()
    for ss in body.split('; '):
        if len(ss.split('=')) != 2:
            print(ss)
            continue
        key, val = ss.split('=')
        items[key.strip()] = val.strip()

    return items


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


def trans_VSCODE(vscode):
    mapping = {
        'v01': 'sc', 'v02': 'scmri', 'v03': 'bl', 'v04': 'm03', 'v05': 'm06', 'v11': 'm12', 'v12': 'm18',
        'v21': 'm24', 'v22': 'm30', 'v31': 'm36', 'v32': 'm42', 'v41': 'm48', 'v42': 'm54', 'v51': 'm60', 'v52': 'm66'
    }
    if vscode in mapping:
        return mapping[vscode]
    else:
        return None


if __name__ == '__main__':
    obj = CollectSamples()
    obj.dcm2nii()
    obj.CreateLink()
    obj.AddShape()
    obj.AddXML()

