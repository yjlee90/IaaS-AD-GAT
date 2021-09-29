import numpy as np
import pandas as pd
import pickle
import os
import torch
from torch._C import device
import torch.nn as nn
from torch.nn.modules.rnn import LSTM
from torch.utils.data import DataLoader
import easydict
import logging

from my_dataset import WorkloadDataset, gen_random_edge_pair

# 로그 생성
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] [%(filename)s - %(funcName)s : %(lineno)d] - %(message)s')

# log 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

num_host = 66
num_feature = 4

args = easydict.EasyDict({
    "batch_size": 1000, ## 배치 사이즈 설정
    "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), ## GPU 사용 여부 설정

    # lstm-encoder
    "input_size": num_host * num_feature, ## 입력 차원 설정
    "hidden_size": 200, ## Hidden 차원 설정
    "num_layers": 1,     ## LSTM layer 갯수 설정

    "output_size": num_host * num_feature, ## 출력 차원 설정

    "learning_rate" : 0.01, ## learning rate 설정
    "max_iter" : 100, ## 총 반복 횟수 설정
    "n_features" : 4,
    "window_size" : 10
})
workload_dataset = WorkloadDataset('./data/mock/processed/') # x = (36000, 66, 4) (total_time, nodes, features)

rand_edge_index = gen_random_edge_pair(66,3000,32)
rand_edge_index = rand_edge_index.to(args.device)

train_size = int(0.8 * len(workload_dataset)) # 28800
test_size = len(workload_dataset) - train_size # 7200
train_dataset, test_dataset = torch.utils.data.random_split(workload_dataset, [train_size, test_size])

train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size)
test_loader = DataLoader(dataset = test_dataset, batch_size = args.batch_size)

#%%
# %%

from torch_geometric.nn import GATConv
import torch.nn.functional as F
class GATNetwork(nn.Module) :
    def __init__(self, n_features) :
        super(GATNetwork,self).__init__()
        self.n_features = n_features

        self.gat1 = GATConv(
            in_channels=self.n_features,
            out_channels=8,
            heads=8,
            dropout=0.6
        )

        self.gat2 = GATConv(
            in_channels = 8*8,
            out_channels= self.n_features,
            heads=1,
            dropout=0.6
        )

    # train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="training")
    def forward(self, x, edge_index) :
        sequence_length, n_node, n_feature = x.size() # sequence_length = batch_size
        logger.debug(f'GAT input x shape {sequence_length, n_node, n_feature}')
        hidden = torch.empty(sequence_length, n_node, n_feature).cuda()

        for t in range(sequence_length) :
            y = F.dropout(x[t], p=0.6, training=self.training)
            y = F.elu(self.gat1(y, edge_index))
            y = F.dropout(y, p=0.6, training=self.training)
            y = self.gat2(y, edge_index)
            result = F.log_softmax(y, dim=-1)
            hidden[t] = result
            
        return hidden # batchsize, num_node(server), num_feature(cpu,memory)
            
class LSTMEncoder(nn.Module) :
        
    def __init__(self, n_features, hidden_size=1024, num_layers=2):
        super(LSTMEncoder, self).__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size = self.n_features,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True,
            dropout = 0.1,
            bidirectional = False
        )

    def forward(self, x):
        "서버 별 순차적으로 해야함 - for loop"
        "Input x : (Batch sequence, Num_node, Num_feature"
        " x = permute(1,0,2) # (Node, Batch_sequence, Num_feature)"
        " xk"
        " LSTM에서 batch를 Node의 갯수로 본다."
        "for i in range(Num node)"
        "   node = x"
        "   (sequence, args.window_size, Num_feature)"
        "   lstm(node)"
        "변경 한다. seq_length는 arg window 사이즈로 받는다"
        x = x.permute(1,0,2) # (node, batch, feature)

        # x: tensor of shape (batch_size, seq_length, hidden_size)
        logger.debug(f'LSTM Encoder Input shape : {x.shape}')

        outputs, (hidden, cell) = self.lstm(x)

        return (hidden, cell)

class LSTMDecoder(nn.Module) :
    
    def __init__(self):
        super().__init__()


class MyModel(nn.Module) :
    def __init__(self, args):
        super(MyModel, self).__init__()

        self.n_features = args.n_features
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers

        self.GATLayer = GATNetwork(
            n_features= self.n_features
        )

        self.LSTMEncoder = LSTMEncoder(
            n_features = self.n_features,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers
        )
        
    def forward(self, x, edge_index):
        sequence_length, n_node, n_feature = x.size() # sequence_length = batch_size

        h_gat = self.GATLayer(x, edge_index)
        logger.info(f'rangd_edge_index tensor 위치 : {h_gat.is_cuda}')
        h_enc = self.LSTMEncoder(h_gat)

        return h_gat


from tqdm import tqdm
model = MyModel(args)
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
epochs = tqdm(range(args.max_iter//len(train_loader)+1))
for epoch in epochs:

    logger.info(f'Batch iterator count : {epoch}/{len(epochs)}')
    model.train()
    optimizer.zero_grad()
    # train_iterator = enumerate(train_loader), total=len(train_loader), desc="training")

    batch_len = len(train_loader)
    count = 0

    for batch_data in train_loader:
        if count == 1 :
            break;
        past_data = batch_data.float().to(args.device)
        rand_edge_index = rand_edge_index.to(args.device)
        out = model(past_data, rand_edge_index)
        count+=1

    if count == 1 :
        break;