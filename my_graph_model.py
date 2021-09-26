# %%
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvLSTM

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

from torch.utils.data import DataLoader
import easydict
from my_dataset import *

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

loader = ChickenpoxDatasetLoader()

dataset = loader.get_dataset()

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

# %%

num_host = 66
num_feature = 4

args = easydict.EasyDict({
    "batch_size": 100, ## 배치 사이즈 설정
    "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), ## GPU 사용 여부 설정
    "input_size": num_host * num_feature, ## 입력 차원 설정
    "hidden_size": 2000, ## Hidden 차원 설정
    "output_size": num_host * num_feature, ## 출력 차원 설정
    "num_layers": 1,     ## LSTM layer 갯수 설정
    "learning_rate" : 0.01, ## learning rate 설정
    "max_iter" : 1000, ## 총 반복 횟수 설정,
    "window_size" : 50,
    "dropout" : 0.1,
    "alpha": 0.2,
    "n_features": num_feature,
    "num_host": num_host
})

# %%
# Data setting 
workload_dataset = SlidingWindowDataset('./data/processed/', args.window_size)
workload_dataset.get_info()

train_size = int(0.8 * len(workload_dataset)) # 28800
test_size = len(workload_dataset) - train_size # 7200
train_dataset, test_dataset = torch.utils.data.random_split(workload_dataset, [train_size, test_size])

train_loader = DataLoader(
                    dataset = train_dataset,
                    batch_size = args.batch_size,
                )
test_loader = DataLoader(
                    dataset = test_dataset,
                    batch_size = args.batch_size,
                )




train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvLSTM(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0
        
model = RecurrentGCN(node_features=4)

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
for time, snapshot in enumerate(test_dataset):
    y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))