import os
import shutil
import json
import pandas as pd
import numpy as np
from datetime import datetime
import nibabel as nib
from joblib import Parallel, delayed
# from concurrent import futures
from subprocess import run
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.use('Agg')


class Pipeline:
    def __init__(self):
        self.method = 'nc'
        self.raw_folder = 'MRIProcess/NACC/raw/'  # raw nifti files
        self.root_folder = 'MRIProcess/NACC/'
        self.out_folder = os.path.join(self.root_folder, f'{self.method}_MNI_brain/')  # processed nii.gz files
        self.image_folder = self.out_folder.replace('MNI_brain', 'image')  # images for QC
        self.polish_folder = self.out_folder.replace('MNI_brain', 'polish')  # images need to be polished
        self.good_folder = self.out_folder.replace('MNI_brain', 'good')  # images with good quality
        self.storage_folder = self.out_folder.replace('MNI_brain', 'storage')  # images with different parameters

        for f in [self.out_folder, self.image_folder, self.polish_folder, self.good_folder, self.storage_folder]:
            os.makedirs(f, exist_ok=True)

        self.raw_list = self.get_raw()  # get all raw files to be processed

    def Initial(self):
        processed_list = find_cases(self.out_folder, '.nii.gz')
        print('Number of ALL cases: ', len(self.raw_list))
        print('Number of processed cases: ', len(processed_list))
        if len(self.raw_list) == len(processed_list):
            print('all cases have been processed')
            return
        file_list = [kk for kk in self.raw_list if kk not in processed_list]  # needs to be processed

        # default parameters
        step4 = '-R -f 0.5 -g 0.0'
        step7 = '-R -f 0.3 -g 0.0'
        params_str = f"{step4}; {step7}"
        SetParams = {kk: params_str for kk in file_list}
        if not os.path.isfile(f'bet_params_{self.method}.json'):
            assert len(processed_list) == 0
            json.dump(SetParams, open(f'bet_params_{self.method}.json', 'w'), indent=2, sort_keys=True)
        else:
            OriginalParams = json.load(open(f'bet_params_{self.method}.json', 'r'))
            OriginalParams.update(SetParams)
            json.dump(OriginalParams, open(f'bet_params_{self.method}.json', 'w'), indent=2, sort_keys=True)

        print(f'[{datetime.now()}] Start processing {len(file_list)} cases.')
        Parallel(n_jobs=10)(delayed(self.once)(name, *SetParams[name].split('; '), polish=-1)
                            for name in tqdm(file_list))
        # run_queue = {name: SetParams[name].split('; ') for name in file_list}
        # try:
        #     with futures.ProcessPoolExecutor(max_workers=10) as executor:
        #         results = list(tqdm(executor.map(self.once, run_queue.items(), [-1] * len(run_queue)),
        #                             total=len(run_queue)))
        # except KeyboardInterrupt:
        #     raise KeyboardInterrupt('terminating all processes.')
        print(f'[{datetime.now()}] Finish.')

    def Polish(self, polish=None):
        OriginalParams = json.load(open(f'bet_params_{self.method}.json', 'r'))
        if polish == 0:
            SetParams = json.load(open(f'bet_params_{self.method}_Setting.json', 'r'))
            in_polish = find_cases(self.polish_folder, '.jpg', suffix='(-R')
            in_json = list(SetParams.keys())
            assert set(in_polish) == set(in_json) and sorted(in_polish) == sorted(in_json)
        else:
            raise ValueError(polish)

        jpg_list = [find_cases(self.out_folder.replace('MNI_brain', kk), '.jpg', suffix='(-R')
                    for kk in ['good', 'bad', 'image', 'polish']]
        jpg_list = [item for sublist in jpg_list for item in sublist]
        if len(jpg_list) != len(self.raw_list):
            print(len(self.raw_list), len(jpg_list), len(self.raw_list) - len(jpg_list))
            # raise AssertionError(f'some cases have not been processed\n'
            #                      f'Number of jpg files: {len(jpg_list)} != total number : {len(self.raw_list)}\n'
            #                      f'{set(self.raw_list) - set(jpg_list)}')

        jpg_list = [find_cases(self.out_folder.replace('MNI_brain', kk), '.jpg')
                    for kk in ['good', 'bad', 'image', 'polish']]
        jpg_list = [item for sublist in jpg_list for item in sublist]
        if set(jpg_list) - set(find_cases(self.storage_folder, '.jpg')) != set():
            raise AssertionError('some cases have not been saved in storage')

        if set(SetParams.keys()) - set(OriginalParams.keys()) != set():
            raise AssertionError('cases in SetParams are not in OriginalParams')

        run_queue = dict()
        for name, param in SetParams.items():
            assert param != OriginalParams[name]
            run_queue[name] = param.split('; ')
        print(f'[{datetime.now()}] Start processing {len(run_queue)} cases.')

        Parallel(n_jobs=10)(delayed(self.once)(name, param[0], param[1], polish)
                            for name, param in tqdm(run_queue.items()))
        # try:
        #     with futures.ProcessPoolExecutor(max_workers=10) as executor:
        #         results = list(tqdm(executor.map(self.once, run_queue.items(), [polish] * len(run_queue)),
        #                             total=len(run_queue)))
        # except KeyboardInterrupt:
        #     raise KeyboardInterrupt('terminating all processes.')

        OriginalParams.update(SetParams)
        json.dump(OriginalParams, open(f'bet_params_{self.method}.json', 'w'), indent=2, sort_keys=True)

        if polish == 0:
            alljson = json.load(open(f'bet_params_{self.method}.json', 'r'))
            setjson = json.load(open(f'bet_params_{self.method}_Setting.json', 'r'))
            for kk, vv in setjson.items():
                assert kk in alljson and vv == alljson[kk]
        print(f'[{datetime.now()}] Finish.')

    def get_raw(self):
        # return find_cases(self.raw_folder, '.nii')
        raw = pd.read_excel('/home/data2/PublicData/NACC/NACC_DataWash.xlsx', sheet_name='RawPath')
        raw = raw.query('UseS2 == 1')
        return [kk.split('.nii')[0] for kk in raw['link']]

    def once(self, name, step4, step7, polish):
        params_str = f"{step4}; {step7}"

        if self.method in ['nc', 'N3nc']:
            retcode = run(' '.join([f'bash pipeline_{self.method}.sh', self.raw_folder, name, self.out_folder,
                          f'"{step4}"', f'"{step7}"']),
                          shell=True, capture_output=True)
        elif self.method == 'ukbb':
            retcode = run(' '.join([f'bash pipeline_{self.method}.sh', self.raw_folder, name, self.out_folder]),
                          shell=True, capture_output=True)
        else:
            raise ValueError(self.method)

        file_name = f'{name}({params_str}).jpg'
        if retcode.stderr.decode('utf-8') == '':
            show_nifti(self.out_folder, self.image_folder, name, params_str)
            assert os.path.isfile(os.path.join(self.image_folder, file_name))
        else:
            print(retcode)
            remove_folder(os.path.join(self.out_folder, name))

        if polish == 0:
            polish_file = [kk for kk in os.listdir(self.polish_folder) if kk.startswith(name)]
            assert len(polish_file) == 1
            assert polish_file[0].split('(')[1].split(')')[0] != params_str
            os.remove(os.path.join(self.polish_folder, polish_file[0]))

        elif polish == 1:
            rm_file = []
            for ff in ['good', 'bad', 'image', 'polish']:
                path = self.image_folder.replace('image', ff)
                polish_file = [kk for kk in os.listdir(path) if
                               kk.startswith(name) and kk.split('(')[1].split(')')[0] != params_str]
                if len(polish_file) == 1:
                    rm_file.append(os.path.join(path, polish_file[0]))
            # print(rm_file)
            assert len(rm_file) == 1
            os.remove(rm_file[0])

        elif polish == 2:
            polish_file = [kk for kk in os.listdir(self.image_folder) if kk.startswith(name)]
            assert len(polish_file) == 2
            for pp in polish_file:
                if pp.split('(')[1].split(')')[0] != params_str:
                    os.remove(os.path.join(self.image_folder, pp))

        if os.path.isfile(os.path.join(self.storage_folder, file_name)):
            shutil.copy2(os.path.join(self.image_folder, file_name), self.storage_folder)
            shutil.move(os.path.join(self.image_folder, file_name), self.good_folder)
        else:
            shutil.copy2(os.path.join(self.image_folder, file_name), self.storage_folder)

    def crop(self):
        path = self.image_folder.replace('image', 'crop')
        if os.path.isfile(os.path.join(path, f'bet_params_{self.method}_crop.json')):
            OriginalParams = json.load(open(os.path.join(path, f'bet_params_{self.method}_crop.json'), 'r'))
        else:
            OriginalParams = dict()
        run_queue = dict()
        for file in sorted(os.listdir(path)):
            if not file.endswith('.jpg') or file.count(';') == 2:
                continue
            subID = file.split('(-R')[0]
            if subID in OriginalParams.keys():
                params_str = OriginalParams[subID]
            else:
                params = file.split('(')[1].split(').jpg')[0]
                crop_params = open(os.path.join(path, f'{subID}_crop.txt'), 'r').readlines()[0].strip('\n')
                params_str = params + '; ' + crop_params
                OriginalParams[subID] = params_str
            if not os.path.isfile(os.path.join(path, f'{subID}({params_str}).jpg')):
                # print(f'{subID}({params_str}).jpg')
                run_queue[subID] = params_str
        for file in sorted(os.listdir(path)):
            if not file.endswith('.jpg') or file.count(';') == 1:
                continue
            subID = file.split('(-R')[0]
            assert subID in OriginalParams.keys()
            params_str = OriginalParams[subID]
            if not os.path.isfile(os.path.join(path, f'{subID}({params_str}).jpg')):
                # print(f'{subID}({params_str}).jpg')
                run_queue[subID] = params_str
        crop_txt = [kk.split('_crop.txt')[0] for kk in os.listdir(path) if kk.endswith('_crop.txt')]
        assert set(crop_txt) == set(OriginalParams.keys())
        print('Number of all cases for crop: ', len(OriginalParams))
        print(f'[{datetime.now()}] Start processing {len(run_queue)} cases.')

        Parallel(n_jobs=10)(delayed(self.once1)(item) for item in tqdm(run_queue.items()))
        # try:
        #     with futures.ProcessPoolExecutor(max_workers=10) as executor:
        #         results = list(tqdm(executor.map(self.once1, run_queue.items()), total=len(run_queue)))
        # except KeyboardInterrupt:
        #     raise KeyboardInterrupt('terminating all processes.')
        json.dump(OriginalParams, open(os.path.join(path, f'bet_params_{self.method}_crop.json'), 'w'),
                  indent=2, sort_keys=True)
        json.dump(run_queue, open(os.path.join(path, 'run_queue.json'), 'w'),
                  indent=2, sort_keys=True)
        print(f'[{datetime.now()}] Finish.')

    def once1(self, item):
        name, params_str = item
        out_folder = '/home/data2/MRIProcess/NACC/nc_crop'
        assert params_str.count(';') == 2
        step4, step7, crop_params = params_str.split('; ')
        retcode = run(' '.join([f'bash pipeline_crop_nc.sh', self.raw_folder, name, out_folder,
                                f'"{step4}"', f'"{step7}"', f'"{crop_params}"']),
                      shell=True, capture_output=True)

        file_name = f'{name}({params_str}).jpg'
        if retcode.stderr.decode('utf-8') == '':
            show_nifti(out_folder, out_folder, name, params_str)
            assert os.path.isfile(os.path.join(out_folder, file_name))
        else:
            print(retcode)
            remove_folder(os.path.join(out_folder, name))

        shutil.copy2(os.path.join(out_folder, file_name), self.storage_folder)
        polish_file = [kk for kk in os.listdir(out_folder) if (kk.startswith(name) and kk.endswith('.jpg'))]
        assert len(polish_file) == 2
        for pp in polish_file:
            if pp.split('(')[1].split(')')[0] != params_str:
                os.remove(os.path.join(self.image_folder, pp))

    def tmp_create_json(self):
        create_json = dict()

        path = self.image_folder.replace('image', 'crop')
        for file in sorted(os.listdir(path)):
            if not file.endswith('.jpg') or file.count(';') == 1:
                continue
            subID = file.split('(-R')[0]
            param_str = file.split('(')[1].split(').jpg')[0]
            create_json[subID] = param_str
        crop_txt = [kk.split('_crop.txt')[0] for kk in os.listdir(path) if kk.endswith('_crop.txt')]
        assert len(create_json) == len(crop_txt)
        json.dump(create_json, open(os.path.join(path, f'bet_params_{self.method}_crop.json'), 'w'),
                  indent=2, sort_keys=True)


class BiasCorr:
    def __init__(self, pre, method, norm_param):
        self.pre = pre
        self.method = method
        self.norm_param = norm_param
        self.root_folder = 'MRIProcess/NACC/'
        self.raw_folder = os.path.join(self.root_folder, f'{pre}_MNI_brain/')  # pre-processed nii.gz files
        self.out_folder = os.path.join(self.root_folder, f'{pre}_{method}_final')  # final output
        self.image_folder = os.path.join(self.root_folder, f'{pre}_{method}_image/')  # images for QC
        os.makedirs(self.out_folder, exist_ok=True)
        os.makedirs(self.image_folder, exist_ok=True)

        self.raw_list = self.get_raw()  # get all raw files to be processed

    def get_raw(self):
        # return find_cases(self.raw_folder, '.nii')
        raw = pd.read_excel('PublicData/NACC/NACC_DataWash.xlsx', sheet_name='RawPath')
        raw = raw.query('Note == 1')  # total=6747, bad=227, Use=6747-227=6520
        return [kk.split('.nii')[0] for kk in raw['link']]

    def Initial(self):
        processed_list = find_cases(self.out_folder, '.nii.gz')
        print('Number of ALL cases: ', len(self.raw_list))
        print('Number of processed cases: ', len(processed_list))
        if len(self.raw_list) == len(processed_list):
            print('all cases have been processed')
            return
        file_list = [kk for kk in self.raw_list if kk not in processed_list]  # needs to be processed

        print(f'[{datetime.now()}] Start processing unprocessed {len(file_list)} cases.')
        Parallel(n_jobs=10)(delayed(self.once)(name) for name in tqdm(sorted(file_list)))
        # for name in file_list[:2]:
        #     self.once(name)
        print(f'[{datetime.now()}] Finish.')

    def plot(self):
        processed_list = find_cases(self.out_folder, '.nii.gz')
        print('Number of ALL cases: ', len(self.raw_list))
        print('Number of processed cases: ', len(processed_list))

        print(f'[{datetime.now()}] Start plotting processed {len(processed_list)} cases.')
        Parallel(n_jobs=10)(delayed(self.once)(name) for name in tqdm(sorted(processed_list)))
        # for name in sorted(processed_list)[:2]:
        #     self.once(name)
        print(f'[{datetime.now()}] Finish.')

    def once(self, name):
        if not os.path.isfile(os.path.join(self.out_folder, f'{name}.nii.gz')):
            retcode = run(' '.join(['bash BiasFieldCorrection.sh', self.raw_folder, name, self.out_folder, self.method]),
                          shell=True, capture_output=True)
            if not retcode.stderr.decode('utf-8') == '':
                print(retcode)
                remove_folder(os.path.join(self.out_folder, name))
                return
        show_nifti_npy(self.out_folder, self.image_folder, name, params=self.norm_param)


# ############# functions #############


def getID(filename):
    if '.nii' in filename:
        return filename.split('.nii')[0]
    elif '.jpg' in filename:
        return filename.split('.jpg')[0]


def find_cases(folder, extension, suffix=None):
    file_list = [getID(file) for file in os.listdir(folder) if file.endswith(extension)]
    if suffix:
        file_list = [file.split(suffix)[0] for file in file_list if suffix in file]
    return file_list


def remove_folder(path):
    if os.path.exists(path):
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        else:
            for filename in os.listdir(path):
                remove_folder(os.path.join(path, filename))
            os.rmdir(path)


def show_nifti(out_folder, image_folder, name, params=''):
    file_name = f'{name}({params}).jpg'

    data = nib.load(os.path.join(out_folder, f'{name}.nii.gz')).get_fdata()
    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(data[int(data.shape[0] / 2), :, :])
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(data[:, int(data.shape[1] / 2), :])
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(data[:, :, int(data.shape[2] / 2)])
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(data[int(data.shape[0] / 4), :, :])
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(data[:, int(data.shape[1] / 4), :])
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(data[:, :, int(data.shape[2] / 4)])
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.88, wspace=0, hspace=0)
    plt.suptitle(f'{name}\nParams: {params}', y=0.98)
    # plt.tight_layout()

    plt.savefig(os.path.join(image_folder, file_name), pad_inches=0)
    plt.close()


def show_nifti_npy(out_folder, image_folder, name, params):
    params = params.copy()
    raw = nib.load(os.path.join(out_folder, f'{name}.nii.gz')).get_fdata()
    fig = plt.figure(figsize=(5, 4))
    gs = gridspec.GridSpec(3, 3)

    ax5 = plt.subplot(gs[0, :])
    data = raw[raw != 0].flatten()
    # print(data.max())
    n, bins, patches = ax5.hist(data, bins=100, density=True)  # range=(1e-8, raw.max()),

    if 'thrN' in params.keys():
        thrN, thrPerc, scale, suffix = params.pop('thrN'), params.pop('thrPerc'), params.pop('scale'), params.pop('suffix')
        assert len(params) == 0
        title = f'{name}({thrN:.0e};{thrPerc:g};{scale};{suffix})'
        # if os.path.isfile(os.path.join(image_folder, title + '.jpg')):
        #     return

        cut = get_cut(n, bins, data, thrN, thrPerc)
        ax5.plot([cut, cut], [0, n.max()], 'r--')
        npy = normalize(raw, cut=cut, scale=scale)
        # print(raw.max(), npy.max())

    else:
        suffix = params.pop('suffix')
        assert len(params) == 0
        title = f'{name}({suffix})'
        if os.path.isfile(os.path.join(image_folder, title + '.jpg')):
            return

        npy = raw / np.mean(raw)
        npy = np.clip(npy, 0, 8)

    np.save(os.path.join(out_folder, title + '.npy'), npy.astype(np.float32))

    ax4 = plt.subplot(gs[1, :])
    data = npy[npy != 0].flatten()
    # print(data.max())
    ax4.hist(data, bins=100, density=True)  # range=(1e-8, data.max()),

    ax1 = plt.subplot(gs[2, 0])
    ax1.imshow(npy[int(npy.shape[0] / 2), :, :])
    ax2 = plt.subplot(gs[2, 1])
    ax2.imshow(npy[:, int(npy.shape[1] / 2), :])
    ax3 = plt.subplot(gs[2, 2])
    ax3.imshow(npy[:, :, int(npy.shape[2] / 2)])
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.subplots_adjust(left=0.1, right=0.92, bottom=0, top=0.92, wspace=0, hspace=0.2)
    plt.suptitle(title, y=0.99)
    # plt.tight_layout()

    plt.savefig(os.path.join(image_folder, title + '.jpg'))  # , pad_inches=0
    plt.close()


def get_cut(n, bins, data, thrN, thrPerc):
    interval = len(n) // 20
    # print(interval)
    bins = bins[:-1]
    # print(np.mean(data), np.median(data), np.percentile(data, 95), np.percentile(data, 99))
    # print(np.percentile(n, 10))
    # print(bins[n < n.max() * 0.01])
    cut = np.where((n < n.max() * thrN) & (bins > np.percentile(data, thrPerc)))[0]
    # print(cut, n[cut[0]])
    if len(cut) > interval:
        cut = [cut[i - interval] for i in range(interval, len(cut)) if n[cut[i - interval]] - n[cut[i]] <= n[cut[i]]]
        # print(cut, n[cut[0]])
    else:
        cut = []
    cut = bins[cut[0]] if len(cut) > 0 else data.max()
    return cut


def normalize(data, cut=None, scale='MinMax'):
    if cut is not None:
        data = np.clip(data, 0, cut)
        # data = np.where(data > cut, 0, data)
    if scale == 'MinMax':
        data = (data - data.min()) / (data.max() - data.min())
    elif scale == 'MeanStd':
        data = (data - data.mean()) / data.std()
    else:
        raise ValueError(scale)
    return data


if __name__ == "__main__":
    main = Pipeline()
    main.Initial()
    main.Polish(polish=0)

    # main.crop()

    main = BiasCorr(pre='nc', method='fast', norm_param=dict(suffix='mean;8'))
    main.plot()

