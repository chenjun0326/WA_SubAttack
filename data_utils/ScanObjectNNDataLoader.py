import torch
import numpy as np
import h5py
import warnings
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ScanObjectNNDataLoader(Dataset):
    def __init__(self, root, split, bg=False):

        self.splits = split
        self.folder = 'training' if self.splits in ['train', 'valid'] else 'test'

        self.npoints = 1024

        self.root = root

        if bg:
            print('Use data with background points')
            dir_name = 'main_split'
        else:
            print('Use data without background points')
            dir_name = 'main_split_nobg'
        file_name = '_objectdataset.h5'
        h5_name = '{}/{}/{}'.format(self.root, dir_name, self.folder + file_name)
        with h5py.File(h5_name, mode="r") as f:
            self.data = f['data'][:].astype('float32')
            self.label = f['label'][:].astype('int64')
        print('The size of %s data is %d' % (split, self.data.shape[0]))


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        try:
            import random as rd
            np.random.seed(2022)
            rd.seed(2022)
            torch.manual_seed(2022)
            pc = self.data[index]
            choice = np.random.choice(pc.shape[0], self.npoints, replace=False)
            pc = pc[choice, :]
            pc = pc[:, :3]
            pc = pc_normalize(pc)
        except:
            import random as rd
            np.random.seed(index)
            rd.seed(index)
            torch.manual_seed(index)
            pc = self.data[index]
            choice = np.random.choice(pc.shape[1], self.npoints, replace=False)
            pc = pc[:, choice]
            pc = pc[:, :, :3]
            pc = pc_normalize(pc)

        label_value = self.label[index]
        if isinstance(label_value, np.ndarray):
            if label_value.shape == ():
                label = np.array([label_value]).astype('int64')
            elif label_value.shape == (1,):
                label = label_value.astype('int64')
            else:
                label = np.array([int(label_value.squeeze())]).astype('int64')
        else:
            label = np.array([label_value]).astype('int64')

        return pc, label