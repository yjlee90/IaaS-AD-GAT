# %% 
import easydict
import torch

import numpy as np
import torch
from torch._C import Graph
import torch.nn as nn
import torch.nn.functional as F

import logging
from my_dataset import *

# 로그 생성
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] [%(filename)s - %(funcName)s : %(lineno)d] - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# %%
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# %%
class Encoder(nn.Module):
    
    def __init__(self, input_size=4096, hidden_size=1024, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=0.1, bidirectional=False)

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        outputs, (hidden, cell) = self.lstm(x)

        return (hidden, cell)

class Decoder(nn.Module):
    
    def __init__(self, input_size=4096, hidden_size=1024, output_size=4096, num_layers=2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.1, bidirectional=False)        

        self.fc = nn.Linear(hidden_size, output_size)
   
    def forward(self, x, hidden):
        # x: tensor of shape (batch_size, seq_length, hidden_size
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)


# %%

class MyModel(nn.Module) :

    def __init__(self, args):
        super(MyModel, self).__init__()

        self.gat = GraphAttentionLayer(
            dropout = args.dropout,
            in_features = args.in_features,
            out_features = args.out_features,
            alpha = args.alpha,
            concat = True,
        )
        
        self.lstm_encoder = Encoder(
            input_size= args.input_size,
            hidden_size = args.hidden_size,
            num_layers = args.num_layers
        )

        self.reconstruct_decoder = Decoder(
            input_size=args.input_size,
            output_size=args.output_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        )

        self.predict_decoder = Decoder(
            input_size=args.input_size,
            output_size=args.output_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )
        
        self.criterion = nn.MSELoss()
        
        
    def forward(self, input, adj, target):
        # input shape - (144, 66, 22) (time_sequence, num_nodes, num_features)
        seq_gat_h= []
        for t in range(len(input)) :
            temp_hidden, temp_attention = self.gat(input[t], adj)
            seq_gat_h.append(temp_hidden)

        seq_gat_h = torch.stack(seq_gat_h)
        seq_gat_h = seq_gat_h.view(1,144, 1452)

        encoder_h = self.lstm_encoder(seq_gat_h)
        sequence_length = 144

        # Prediction 
        predict_output = []
        temp_input = torch.zeros((1,1,1452), dtype=torch.float)
        pred_decoder_h = encoder_h

        for t in range(sequence_length):
            temp_input, pred_decoder_h = self.predict_decoder(temp_input, pred_decoder_h)
            predict_output.append(temp_input)
        predict_output = torch.cat(predict_output, dim=1)
        predict_loss = self.criterion(predict_output, target.view(1,144,1452))


        # Reconstruction 
        inverse_idx = torch.arange(144 - 1, -1, -1).long() #[144, 142, 141, 140 ... 1, 0]
        reconstruct_output = []
        temp_input = torch.zeros((1,1,1452), dtype=torch.float)
        recon_deconder_h = encoder_h

        for t in range(sequence_length):
            temp_input, recon_deconder_h = self.reconstruct_decoder(temp_input, recon_deconder_h)
            reconstruct_output.append(temp_input)

        reconstruct_output = torch.cat(reconstruct_output, dim=1)
        reconstruct_loss = self.criterion(reconstruct_output, input.view(1,144,1452)[:, inverse_idx, :])

        return reconstruct_loss, predict_loss



# %%

# %%
"""
Data Prepreration
"""

# from my_dataset import WorkloadDataset, gen_random_edge_pair
# workload_dataset = WorkloadDataset('./data/real/processed/') # x = (36000, 66, 4) (total_time, nodes, features)
# workload_dataset.get_info()

from torch.utils.data import DataLoader, dataloader

adj = torch.randint(2, (66,66))
train_loader = DataLoader(dataset=torch.rand(720, 66,22), batch_size=144, num_workers=8, drop_last=True, shuffle=False)

# %%
"""
Experiment parameter settings
"""

num_node = 66
num_features = 22
args = easydict.EasyDict({
    # 기본 사항
    "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), ## GPU 사용 여부 설정
    "max_iter" : 10, 
    "learning_rate" : 0.001, 


    # GAT 모델 관련
    "in_features" : num_features, # 서버의 feature
    "out_features" : num_features, # 서버의 feature
    "dropout" : 0.6, # 서버의 feature
    "alpha" : 0.1, # 서버의 feature

    # LSTM 모델 관련
    "input_size" : num_node * num_features,
    "hidden_size" : 1000,
    "num_layers" : 1,
    "output_size" : num_node * num_features,
})

# %%
'''
학습하기
'''

model = MyModel(args)
epochs =  range(args.max_iter//len(train_loader) + 1)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

for epoch in epochs:
    logger.debug(f'epochs : {epoch}')
    model.train()
    optimizer.zero_grad()

    train_iterator = iter(train_loader)

    past_seq = next(train_iterator)
    for idx in range(len(train_iterator) - 1) :
        logger.debug(f'batch count : {idx}/{len(train_iterator)-1}')

        future_seq = next(train_iterator)
        logger.debug(f'past_seq: {past_seq.shape}')
        logger.debug(f'future_seq: {future_seq.shape}')


        reconstruct_loss, predict_loss = model(past_seq, adj, future_seq)

        ## Composite Loss
        loss = reconstruct_loss + predict_loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.debug(f'train reconsruction loss: {reconstruct_loss}')
        logger.debug(f'train predition loss: {predict_loss}')

        logger.debug(f'train loss: {loss}')

        past_seq = future_seq
        # train_iterator.set_postfix({
        #     "train_loss": float(loss),
        # })


    # model.eval()
    # eval_loss = 0
    # test_iterator = tqdm(enumerate(test_loader), total=len(test_loader), desc="testing")
    # with torch.no_grad():
    #     for i, batch_data in test_iterator:
    #         future_data, past_data = batch_data

    #         ## 데이터 GPU 설정 및 사이즈 조절
    #         batch_size = past_data.size(0)
    #         example_size = past_data.size(1)
    #         past_data = past_data.view(batch_size, example_size, -1).float().to(args.device)
    #         future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)

    #         reconstruct_loss, predict_loss = model(past_data, future_data)

    #         ## Composite Loss
    #         loss = reconstruct_loss + predict_loss

    #         eval_loss += loss.mean().item()

    #         test_iterator.set_postfix({
    #             "eval_loss": float(loss),
    #         })
    # eval_loss = eval_loss / len(test_loader)
    # print("Evaluation Score : [{}]".format(eval_loss))


# %%

# %%

# %%
