import pickle
import os
import torch
from torch.utils.data import  Dataset

class WorkloadDataset(Dataset) :
    def __init__(self, data_path):
        self._get_all_workload_data(data_path)

    def _get_all_workload_data(self, data_path) :
        file_list = [host for host in os.listdir(data_path) if host.endswith('pkl')]
        self.dataset = [] 
        for file in file_list : 
            with open(f'{data_path}/{file}', 'rb') as f:
                data = pickle.load(f)
            data = torch.from_numpy(data.values).float()
            self.dataset.append(data)
        self.dataset = torch.stack(self.dataset)
        self.dataset = self.dataset.permute(1,0,2) # (time, node, feature)
        # self.dataset = torch.permute(self.dataset, [1,0,2]) # (time, node, feature) torch 1.9.1

    def __getitem__(self,index):
        # node_oriented = torch.permute(1,0,2) # (node,time,feature)
        return self.dataset[index]

    def __len__(self) :
        return len(self.dataset) 
        
class SlidingWindowDataset(Dataset) :
    def __init__(self, data_path, window, target_dim=None, horizon=1):
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon
        self._get_all_workload_data(data_path)
        
    def _get_all_workload_data(self, data_path) :
        file_list = [host for host in os.listdir(data_path) if host.endswith('pkl')]
        list_host_wl_data = [] 
        for file in file_list : 
            with open(f'{data_path}/{file}', 'rb') as f:
                host_wl_data = pickle.load(f)
            host_wl_data = torch.from_numpy(host_wl_data.values).float()
            list_host_wl_data.append(host_wl_data)

        list_host_wl_data = torch.stack(list_host_wl_data)
        list_host_wl_data = list_host_wl_data.permute(1,0,2) # (time, node, feature)
        # list_host_wl_data = torch.permute(list_host_wl_data, [1,0,2]) # (time, node, feature) torch 1.9.1

        self.dataset = [] 

        for slide in range(len(list_host_wl_data) - self.window + 1) :
            x = list_host_wl_data[slide: slide + self.window]
            self.dataset.append(x)
        self.dataset = torch.stack(self.dataset)


    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def get_info(self):
        print('=================workload_dataset meta==================')
        print(f'workload_dataset     window_size : {self.window}')
        print(f'workload_dataset             len : {len(self.dataset)}')
        print(f'workload_dataset each data shape : {self.dataset.data[0].shape}')
        


if __name__ == "__main__":
    window_size = 50
    workload_dataset = SlidingWindowDataset('./data/processed/', window_size)
    workload_dataset.get_info()
