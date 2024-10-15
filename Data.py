import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class MyData(Dataset):
    def __init__(self, data_path, split='train', type='all'):
        self.split = split
        self.type = type
        if self.split == 'own':
            self.data_path = data_path.replace('AbDab', 'Wang')
            self.data = pd.read_csv(os.path.join(self.data_path, 'test_{}.csv'.format(self.type)))
        else:
            self.data_path = data_path
            self.data = pd.read_csv(os.path.join(self.data_path, '{}.csv'.format(self.split)))
        self.label = self.data['Label'].to_numpy()        
        self.FVL = self.data['FV_L'].to_numpy()
        self.FVH = self.data['FV_H'].to_numpy()
        self.virus_rbd = self.data['ags_RBD'].to_numpy()
        self.len = len(self.label)

    def __getitem__(self, index):
        return self.label[index], self.virus_rbd[index], self.FVH[index], self.FVL[index]

    def __len__(self):
        return self.len
    
