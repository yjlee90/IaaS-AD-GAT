# %%
from datetime import datetime
from os import altsep
from pandas.io.pytables import dropna_doc
import torch.nn as nn
from torch.autograd import Variable
from my_dataloader import *
from my_model import MyModel

import pandas as pd

# %%
# node_dataset = NodeDataset('./yjlee-nn/data/processed/')
node_dataset = NodeDataset('./data/processed/')
node_dataloader = DataLoader(dataset=node_dataset, batch_size=8, shuffle=False)

# for i in range(len(node_dataset)) :
#     sample = node_dataset[i]
#     window_sample = SlidingWindowDataset(data=sample, window=10) 
#     for j in range(10):
#         print(window_sample[j])

# %%
vm1 = pd.read_pickle('./data/processed/bigdata-elasticsearch-data-1.pkl').astype(np.float32)
vm2 = pd.read_pickle('./data/processed/bigdata-elasticsearch-data-2.pkl').astype(np.float32)
vm3 = pd.read_pickle('./data/processed/bigdata-elasticsearch-data-3.pkl').astype(np.float32)

vm1 = vm1.mean_cpu_usage
vm2 = vm2.mean_cpu_usage
vm3 = vm3.mean_cpu_usage

total_vm = pd.concat([vm1, vm2,vm3], axis=1)

tensor_total_vm = torch.tensor(total_vm.values)
tensor_total_vm = torch.reshape(tensor_total_vm, [15, 2557, 3])

# %%
n_features = 3 # number of VMs
batch = 100
window_size = 2557
dropout = 0.3
alpha = 0.2

# %%
model = MyModel(
    n_features = n_features,
    window_size = window_size,
    dropout= dropout,
    alpha= alpha,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
result = model(tensor_total_vm)
result

# %%

# %%
