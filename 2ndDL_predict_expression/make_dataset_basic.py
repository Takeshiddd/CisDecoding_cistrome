

import re
import numpy as np 
from collections import defaultdict
import os 
import math
import argparse
from glob import glob
from tqdm import tqdm
import pandas as pd 
from glob import glob
from time import time
import sys
import shutil


def get_sub_dir_idx(i, data_size):
    quot = math.ceil(data_size / 10)
    idx = i // quot
    return idx


def inp(dataset_dir):
    print('************************ WARNING ************************')
    i = 0
    while True:
        i += 1
        try:
            dic={'Yes':True,'yes':True,'YES':True,'No':False,'no':False,'NO':False}
            print('If you start this code, the current dataset \"{}\" will be removed.'.format(dataset_dir))
            inp = dic[input('woud you like to excute？ ----- Yes/No? >> ').lower()]
            break
        except:
            if i >= 5:
                inp = False
                break
            print('')
            pass
        
    if not inp:
        print('Exit.')
        sys.exit()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_root', default='raw_data', type=str, help='path to gene data root.')
    parser.add_argument('--dataset_root', default='gene_dataset', type=str, help='path to dataset root used for training.')
    parser.add_argument('--length', default='20', type=int, help='data length or bin numbers (int)')
    args = parser.parse_args()

    inp(args.dataset_root)

    print('Excute.')

    # Get paths of raw data files.
    gene_data_paths = glob(os.path.join(args.raw_data_root, '*' + '.txt'))

    # Get valid gene that has no missing values.
    print('')
    print('Extracting valid gene that has no missing values.')
    for i, gene_data_path in tqdm(enumerate(gene_data_paths)):
        df = pd.read_csv(gene_data_path, delimiter='\t', index_col=0, usecols=list(range(int(args.length)+1)))
        df = df.dropna()
        gene_names = set(df.index)
        if i == 0:
            valid_names = gene_names
        else:
            valid_names = valid_names & gene_names
        

    # Make dataset folder.
    DATASET_ROOT = args.dataset_root
    SUB_DIRS = ['train_0' + str(tag) for tag in range(10)]
    shutil.rmtree(DATASET_ROOT)
    if not os.path.isdir(DATASET_ROOT):
        os.mkdir(DATASET_ROOT)
    for _dir in SUB_DIRS:
        if not os.path.isdir(os.path.join(DATASET_ROOT, _dir)):
            os.mkdir(os.path.join(DATASET_ROOT, _dir))
    
    # Init dataset files.
    for i, gene_name in enumerate(valid_names):
        sub_dir = SUB_DIRS[get_sub_dir_idx(i, len(valid_names))]
        file_path = os.path.join(DATASET_ROOT, sub_dir, gene_name+'.txt')
        with open(file_path, 'w') as f:
            pass
    
    # Write dataset files.
    print('')
    print('Writing dataset files.')
    for gene_data_path in tqdm(gene_data_paths):
        df = pd.read_csv(gene_data_path, delimiter='\t', index_col=0, usecols=list(range(int(args.length)+1)))
        
        for gene_name in valid_names:
            file_path = glob(os.path.join(DATASET_ROOT, '*', gene_name+'.txt'))[0]
            with open(file_path, 'a') as f:
                ar = np.array([df.loc[gene_name]])
                np.savetxt(f, ar)


    # Translate data format from '.txt' to '.npy'
    print('')
    print('Translating data format from \'.txt\' to \'.npy\'.')
    for sub_dir in SUB_DIRS:
        print(os.path.split(sub_dir)[-1])
        file_paths = glob(os.path.join(DATASET_ROOT, sub_dir, '*'+'.txt'))
        
        for path in tqdm(file_paths):
            export_path = os.path.splitext(path)[0] + '.npy'
            np.save(export_path, np.loadtxt(path).transpose(1, 0))
            os.remove(path)
