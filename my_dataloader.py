# %%
import numpy as np
import pickle
import torch
import os
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

class NodeDataset(Dataset) :
    def __init__(self, data_path):

        self.data_path = data_path
        self.file_list = [host for host in os.listdir(data_path) if host.endswith('pkl')]

    def __getitem__(self,index):
        file = self.file_list[index]
        with open(f'{self.data_path}/{file}', 'rb') as f:
            data = pickle.load(f)
        data = torch.from_numpy(data).float()
        return data

    def __len__(self) :
        return len(self.file_list) 
        
        


class SlidingWindowDataset(Dataset) :
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon
        
    def __getitem__(self, index):
        cols = self.data.columns
        x = torch.tensor(self.data[index : index + self.window][cols].values).to(dtype=torch.double)
        y = torch.tensor(self.data[index : index + self.window + self.horizon][cols].values).to(dtype=torch.double)
        # y = self.data[index + self.window : index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window





def create_data_loaders(
    train_dataset,
    batch_size,
    val_split=0.1,
    shuffle=True,
    test_dataset=None
):
    '''
    :param train_dataset    train_dataset
    :param batch_size       batch_size
    :param val_split        validation split ratio default =0.1
    :param shuffle          using suffle random index
    :param test_dataset     test_dataset
    '''
    train_loader, val_loader, test_loader = None, None, None

    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
    

# %%
