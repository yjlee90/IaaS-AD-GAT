# %%
import numpy as np
import pandas as pd
import pickle
import os
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, dataloader, dataset

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
        self.dataset = torch.permute(self.dataset, [1,0,2]) # (time, node, feature)

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
        list_host_wl_data = torch.permute(list_host_wl_data, [1,0,2]) # (time, node, feature)

        self.dataset = [] 

        for slide in range(len(list_host_wl_data) - self.window + 1) :
            x = list_host_wl_data[slide: slide + self.window]
            self.dataset.append(x)
        self.dataset = torch.stack(self.dataset)


    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

workload_dataset = SlidingWindowDataset('./data/processed/', 50)
for idx, temp_data in enumerate(workload_dataset) :
    if torch.isnan(temp_data).any() :
        print(idx)
        print(temp_data)
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
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)


# class GATEncoder(nn.Module) :




class Seq2Seq(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        hidden_size = args.hidden_size
        input_size = args.input_size
        output_size = args.output_size
        num_layers = args.num_layers
        
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.reconstruct_decoder = Decoder(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        
        self.predict_decoder = Decoder(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        
        self.criterion = nn.MSELoss()
        # self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, src, trg):
        
        batch_size, sequence_length, img_size = src.size()
        
        ## Encoder 넣기
        encoder_hidden = self.encoder(src)
        
        predict_output = []
        temp_input = torch.zeros((batch_size,1,img_size), dtype=torch.float).to(src.device)
        hidden = encoder_hidden
        for t in range(sequence_length):
            temp_input, hidden = self.predict_decoder(temp_input, hidden)
            predict_output.append(temp_input)
        predict_output = torch.cat(predict_output, dim=1)
        predict_loss = self.criterion(predict_output, trg)

        inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
        reconstruct_output = []
        temp_input = torch.zeros((batch_size,1,img_size), dtype=torch.float).to(src.device)
        hidden = encoder_hidden
        for t in range(sequence_length):
            temp_input, hidden = self.reconstruct_decoder(temp_input, hidden)
            reconstruct_output.append(temp_input)
        reconstruct_output = torch.cat(reconstruct_output, dim=1)
        reconstruct_loss = self.criterion(reconstruct_output, src[:, inv_idx, :])
            
        return reconstruct_loss, predict_loss
    
    def generate(self, src):
        batch_size, sequence_length, img_size = src.size()
        
        ## Encoder 넣기
        hidden = self.encoder(src)
        
        outputs = []
        
        temp_input = torch.zeros((batch_size,1,img_size), dtype=torch.float).to(src.device)
        for t in range(sequence_length):
            temp_input, hidden = self.predict_decoder(temp_input, hidden)
            outputs.append(temp_input)
        
        return torch.cat(outputs, dim=1)
    
    def reconstruct(self, src):
        batch_size, sequence_length, img_size = src.size()
        
        ## Encoder 넣기
        hidden = self.encoder(src)
        outputs = []
        
        temp_input = torch.zeros((batch_size,1,img_size), dtype=torch.float).to(src.device)
        for t in range(sequence_length):
            temp_input, hidden = self.reconstruct_decoder(temp_input, hidden)
            outputs.append(temp_input)
        
        return torch.cat(outputs, dim=1)

# %% 
# train, test dataset
# %%
import easydict
num_host = 66
num_feature = 4

args = easydict.EasyDict({
    "batch_size": 200, ## 배치 사이즈 설정
    "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), ## GPU 사용 여부 설정
    "input_size": num_host * num_feature, ## 입력 차원 설정
    "hidden_size": 1000, ## Hidden 차원 설정
    "output_size": num_host * num_feature, ## 출력 차원 설정
    "num_layers": 1,     ## LSTM layer 갯수 설정
    "learning_rate" : 0.005, ## learning rate 설정
    "max_iter" : 1000, ## 총 반복 횟수 설정
})


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

# %%
# training
from tqdm import tqdm
model = Seq2Seq(args)
# optimizer 설정
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
epochs = tqdm(range(args.max_iter//len(train_loader)+1))

count = 0
for epoch in epochs:
    model.train()
    optimizer.zero_grad()
    train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="training")

    count += 1 
    for i, batch_data in train_iterator:
        past_data = batch_data
        future_data = batch_data
        batch_size = past_data.size(0)
        example_size = past_data.size(1)
        past_data = past_data.view(batch_size, example_size, -1).float().to(args.device)
        future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)

        reconstruct_loss, predict_loss = model(past_data, future_data)

        ## Composite Loss
        loss = reconstruct_loss + predict_loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_iterator.set_postfix({
            "train_loss": float(loss),
        })
 
    model.eval()
    eval_loss = 0
    test_iterator = tqdm(enumerate(test_loader), total=len(test_loader), desc="testing")
    with torch.no_grad():
        for i, batch_data in test_iterator:
            future_data = batch_data
            past_data = batch_data

            ## 데이터 GPU 설정 및 사이즈 조절
            batch_size = past_data.size(0)
            example_size = past_data.size(1)
            past_data = past_data.view(batch_size, example_size, -1).float().to(args.device)
            future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)

            reconstruct_loss, predict_loss = model(past_data, future_data)

            ## Composite Loss
            loss = reconstruct_loss + predict_loss

            eval_loss += loss.mean().item()

            test_iterator.set_postfix({
                "eval_loss": float(loss),
            })
    eval_loss = eval_loss / len(test_loader)
    print("Evaluation Score : [{}]".format(eval_loss))


# %%

# %%
