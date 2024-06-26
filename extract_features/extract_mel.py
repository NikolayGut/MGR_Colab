import numpy as np
from pickle import dump
import os
import librosa as lbr
import sys

gap = 661795  # 22050*30 = 661500
gap1 = 22050*10  # 22050*30 = 661500


MEL_KWARGS = {
    'n_fft': 2048,
    'hop_length': 1024,
    'n_mels': 128
}

def linspace_gap1(aa,namea):
    data1 = list()
    if aa.shape[0] < gap:
        line = 'len of {}less than {}*3 with len {}\n'.format(namea, gap1, aa.shape[0])
        print(line)
        if aa.shape[0]> gap//2:
            aa1 = np.zeros(gap)
            diffe = gap -aa.shape[0]
            aa1[:aa.shape[0]] = aa
            aa1[-diffe:] = aa[-diffe:]
            aa = aa1
        else:
            couple = gap//aa.shape[0] + 1
            new_list = list()
            for i in range(couple):
                new_list.append(aa)
            cc = np.concatenate(new_list,axis=0)
            aa = cc[:gap]
    for i in range(0, aa.shape[0], gap1):
        if aa[i:].shape[0] >= gap1:
            data1.append(aa[i:i + gap1])
    new_data = list()
    len_b = len(data1)
    if len_b >= 3:
        new_data.append(data1[len_b//2 - 1])
        new_data.append(data1[len_b // 2])
        new_data.append(data1[len_b // 2 + 1])
        return new_data
    else:
        diff = 3 - len_b
        for i in range(len(data1)):
            new_data.append(data1[i])
        for j in range(diff):
            new_data.append(data1[-1])
        return new_data

def load_melspectrogram(filename):
    print(filename)
    new_input, sample_rate = lbr.load(filename)
    name_1 = filename.split('/')[-1]
    x_slice = linspace_gap1(new_input,name_1)
    new_slice = list()
    avv = 0
    print('python version', sys.version)
    print('librosa versiom', lbr.__version__)
    if len(x_slice) < 3:
        print('{}len less than 3 with len equal {}'.format(name_1, len(x_slice)))
    for u in range(len(x_slice)):
        new_x = x_slice[u]

        features = lbr.feature.melspectrogram(new_x, **MEL_KWARGS).T
        features[features == 0] = 1e-6 
        feat = np.log(features)
        new_slice.append(feat)
        avv = feat.shape
    return new_slice

if __name__ == '__main__':
    root = '/content/fma/data/tracks_folder' 
    dirs = ['Electronic','Experimental','Folk','Hip-Hop','Instrumental','International','Pop','Rock']
    for i in range(len(dirs)):
        if i>-1:
            path_dir = os.path.join(root,dirs[i])
            print(path_dir)
            assert os.path.isdir(path_dir)
            if os.path.isdir(path_dir):
                files = os.listdir(path_dir)
                files.sort()
                files_path = [ os.path.join(path_dir,files[k]) for k in range(len(files))]
                content = dict()
                content[dirs[i]]=dict()
                for j in range(len(files_path)):
                    feat = load_melspectrogram(files_path[j])
                    print(files_path[j],feat[0].shape)
                    content[dirs[i]][files_path[j].split('/')[-1]] = feat
            output_path = '/content/fma_small_part_feat/mel/{}.pkl'
            output_path = output_path.format(dirs[i])
            with open(output_path, 'wb') as f:
                dump(content, f)
