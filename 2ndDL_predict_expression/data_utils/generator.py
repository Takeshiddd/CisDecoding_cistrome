import math
import numpy as np
import os
import csv
from glob import glob
from tensorflow.python.keras.utils import Sequence
from operator import itemgetter




class GeneDataGenerator(Sequence):
    def __init__(self, batch_size, input_data_paths, targets, data_length=40):
        self.batch_size = batch_size
        self.input_data_paths = input_data_paths
        self.targets = targets
        self.data_length = data_length
        self.num_nega_data = self.targets.argmax(-1).sum()
        self.num_posi_data = len(targets) - self.num_nega_data
        
    def __getitem__(self, idx):
        input_paths_batch = self.input_data_paths[idx * self.batch_size: (idx+1) * self.batch_size]
        batch_inputs = np.array([np.load(path) for path in input_paths_batch])[:, -self.data_length:, :]
        batch_targets = self.targets[idx * self.batch_size: (idx+1) * self.batch_size]
        return batch_inputs ,batch_targets
    
    def __len__(self):
        return math.ceil(len(self.input_data_paths) / self.batch_size)

    def on_epoch_end(self):
        pass
    
def _get_target_dict(target_file_path):
    target_dict = {}
    with open(target_file_path) as f:
        reader = csv.reader(f, delimiter='\t', skipinitialspace=True)
        for row in reader:
            target_dict[row[0]] = int(row[1])
    return target_dict

def _shuffle(input_data_paths, targets):
    data = list(zip(input_data_paths, targets))
    np.random.shuffle(data)
    input_data_paths, targets = zip(*data)
    return input_data_paths, targets

def _get_matched_datasets(input_data_paths, target_file_path, val_rate=0, shuffle=True):
    binaly_to_category = [[1, 0], [0, 1]]
    
    
    new_input_data_paths_p = []
    new_input_data_paths_n = []
    targets_p = []
    targets_n = []
    rejected = []
    
    target_dict = _get_target_dict(target_file_path)
    for path in input_data_paths:
        gene_name = os.path.splitext(os.path.basename(path))[0][1:]
        if gene_name in target_dict.keys():
            target_id = target_dict[gene_name]
            if target_id == 0:
                new_input_data_paths_n.append(path)
                targets_n.append(binaly_to_category[target_id])
            else:
                new_input_data_paths_p.append(path)
                targets_p.append(binaly_to_category[target_id])

        else: 
            rejected.append(gene_name)
    
    if len(rejected) != 0:
        print('Rejected genes are following.:\n' + str(rejected))
    
    train_input_data_paths_n, train_targets_n, val_input_data_paths_n, val_targets_n = _split_train_val(new_input_data_paths_n, targets_n, val_rate)
    train_input_data_paths_p, train_targets_p, val_input_data_paths_p, val_targets_p = _split_train_val(new_input_data_paths_p, targets_p, val_rate)


    train_input_data_paths = train_input_data_paths_n + train_input_data_paths_p
    train_targets = train_targets_n + train_targets_p
    val_input_data_paths = val_input_data_paths_n + val_input_data_paths_p
    val_targets = val_targets_n + val_targets_p


    if shuffle:
        train_input_data_paths, train_targets = _shuffle(train_input_data_paths, train_targets)
        
    return train_input_data_paths, np.array(train_targets), val_input_data_paths, np.array(val_targets)


def _split_train_val(input_data_paths, targets, val_rate):
    train_length = int(len(input_data_paths) * (1-val_rate))
    train_input_data_paths = input_data_paths[:train_length]
    train_targets = targets[:train_length]
    val_input_data_paths = input_data_paths[train_length:]
    val_targets = targets[train_length:]
    return train_input_data_paths, train_targets, val_input_data_paths, val_targets

def get_train_val_generator(batch_size, target_file, dataset_root='gene_dataset', data_length=500, val_rate=0, shuffle=True):
    if val_rate < 0  or 1 < val_rate:
        raise ValueError("val_rate should be in range 0~1")

    input_data_paths = glob(os.path.join(dataset_root, '*', '*'))
    target_file_path = os.path.join(target_file)

    train_input_data_paths, train_targets, val_input_data_paths, val_targets  \
        = _get_matched_datasets(input_data_paths, target_file_path, val_rate=val_rate, shuffle=shuffle)


    if len(val_input_data_paths) == 0:
        return GeneDataGenerator(batch_size, input_data_paths, train_targets, data_length=data_length), None, val_targets

    elif len(train_input_data_paths) == 0:
        return None, GeneDataGenerator(batch_size, input_data_paths, val_targets, data_length=data_length), val_targets
    
    else:
        train_gen = GeneDataGenerator(batch_size, train_input_data_paths, train_targets, data_length=data_length)
        valid_gen = GeneDataGenerator(batch_size, val_input_data_paths, val_targets, data_length=data_length)
        return train_gen, valid_gen, val_targets


        


if __name__ == "__main__":
    data_paths = glob(os.path.join('dataset', '*', '*'))
    ar = np.empty((len(data_paths), 40, 2))
    for i, path in enumerate(data_paths):
        ar[i] = np.load(path)
    print("")
