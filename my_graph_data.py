# %%
import numpy as np
import pickle
import torch
import os
from torch.utils import data
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
        
        
# %%
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class WorkloadDatasetLoader(object):
    '''
    input : xxxx.pkl file {metric}-{vm}.pkl
    column : time \ mean_metric_avg
    '''

    def __init__(self, path, edge_index, count=None):
        self._edges = edge_index
        self.count = count
        self._read_files(path)

    def _read_files(self, path) :
        self.file_list = [host for host in os.listdir(path) if host.endswith('pkl')]
        self.dataset = {}

        c = 0
        for file in self.file_list :
            if self.count != None :
                if c > self.count :
                    break;

            hostname = file.rstrip('.pkl')
            self.dataset[hostname] = pd.read_pickle(path+file)
            c += 1

        self.time_period = len(self.dataset[hostname])

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_features_and_target(self):
        self.features = []
        self.targets = []
        for time in tqdm(range(self.time_period)) :
            feature_at_time = [ ]
            target_at_time = [ ]
            for hostname in self.dataset.keys() :
                feature_at_time.append(self.dataset[hostname].iloc[time,[0,2,3,4]].values)
                target_at_time.append(self.dataset[hostname].iloc[time,1])

            self.features.append(feature_at_time)
            self.targets.append(target_at_time)

    def get_dataset(self):
        self._get_features_and_target()
        self._get_edge_weights()
        dataset = StaticGraphTemporalSignal(self._edges, self._edge_weights, self.features, self.targets)
        return dataset

# %%
edge_index = torch.tensor(
   [[0,0,0,1,1,2,2,2,3,3,3,4,4,4],
    [1,2,3,0,4,0,3,4,0,2,4,1,2,3]],
    dtype=torch.long
)

small_edge_index = torch.tensor(
   [[0,0,1,1,2,2],
    [1,2,0,2,0,1]],
    dtype=torch.long
)

dataset = WorkloadDatasetLoader('data/processed/', edge_index=edge_index)
result_dataset = dataset.get_dataset()
# %%
for epoch in tqdm(range(200)):
    cost = 0
    for time, snapshot in enumerate(result_dataset):
        print(time)
        print(snapshot)

# %%

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DyGrEncoder

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

loader = ChickenpoxDatasetLoader()
dataset = loader.get_dataset()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DyGrEncoder(
            conv_out_channels=4,
            conv_num_layers=1,
            conv_aggr="mean",
            lstm_out_channels=32,
            lstm_num_layers=1
        )
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight, h_0, c_0):
        h, h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h_0, c_0)
        h = F.relu(h)
        h = self.linear(h)
        return h, h_0, c_0
        
model = RecurrentGCN(node_features = 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

for epoch in tqdm(range(200)):
    cost = 0
    h, c = None, None

    for time, snapshot in enumerate(train_dataset):
        y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)

    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    
model.eval()
cost = 0
h, c = None, None
for time, snapshot in enumerate(test_dataset):
    y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))






# %%

class RecurrentGatEncoder(torch.nn.Module) :
    def __init__(self, conv_output_channels, lstm_output_channels, lstm_num_layers):
        super(RecurrentGatEncoder, self).__init__()


        # LSTM
        self.conv_output_channels = conv_output_channels
        self.lstm_output_channels = lstm_output_channels
        self.lstm_num_layer = lstm_num_layers

        self.reccurent_layer = torch.nn.modules.LSTM(
            input_size = self.conv_output_channels,
            hidden_size = self.lstm_output_channels,
            num_layers=self.lstm_num_layer
        )



