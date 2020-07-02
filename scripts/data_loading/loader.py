from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class dwellingsDataset(Dataset):
    
    def __init__(self, features_dir, labels_dir):
        '''
        '''
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        n = 0
        for file in os.listdir(features_dir):
            if file.endswith('.npy'):
                n += 1
        self.total_n = n

    def __len__(self):
        return self.total_n

    def __getitem__(self, idx):

        features_file = self.features_dir + '/' + 'kam_train_' + str(idx) + '.npy'
        labels_file = self.labels_dir + '/' + 'kam_label_' + str(idx) + '.npy'
        features = np.load(features_file)
        labels = np.load(labels_file)

        return features, labels