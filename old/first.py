# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.manual_seed(1)
# %%

import pandas as pd
# file_path = "https://raw.githubusercontent.com/robertoamansur/rare_event_pred_maintanance/master/processminer-rare-event-mts%20-%20data.csv"
# df = pd.read_csv(file_path, sep=";")
# df.to_feather("./data/processminer-rare-event-mts.ftr")
df = pd.read_feather("./data/processminer-rare-event-mts.ftr")
df
# %%
sign = lambda x: (1, -1)[x < 0]
def curve_shift(df, shift_by):
    vector = df['y'].copy()
    for s in range(abs(shift_by)):
        tmp = vector.shift(sign(shift_by))
        tmp = tmp.fillna(0)
        vector += tmp
    labelcol = 'y'
    df.insert(loc=0, column=labelcol+'tmp', value=vector)
    df = df.drop(df[df[labelcol] == 1].index)
    df = df.drop(labelcol, axis=1)
    df = df.rename(columns={labelcol+'tmp': labelcol})
    df.loc[df[labelcol] > 0, labelcol] = 1
    return df
from copy import deepcopy
df_ = deepcopy(df)
shifted_df = curve_shift(df_, shift_by=-2)
# %%
# drop remove columns
shifted_df = shifted_df.drop(['time','x28','x61'], axis=1)

# x, y
input_x = shifted_df.drop('y', axis=1).values
input_y = shifted_df['y'].values

n_features = input_x.shape[1]
# %%
def temporalize(X, y, timesteps):
    output_X = []
    output_y = []
    for i in range(len(X) - timesteps - 1):
        t = []
        for j in range(1, timesteps + 1):
            # Gather the past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
        output_X.append(t)
        output_y.append(y[i + timesteps + 1])
    return np.squeeze(np.array(output_X)), np.array(output_y)
timesteps = 5
# Temporalize
x, y = temporalize(input_x, input_y, timesteps)
print(x.shape) # (18268, 5, 59)
# %%
# Split into train, valid, and test 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

print(len(x_train))  # 11691
print(len(x_valid))  # 2923
print(len(x_test))   # 3654
#%%
# For training the autoencoder, split 0 / 1
x_train_y0 = x_train[y_train == 0]
x_train_y1 = x_train[y_train == 1]

x_valid_y0 = x_valid[y_valid == 0]
x_valid_y1 = x_valid[y_valid == 1]
# %%
def flatten(x) :
    num_instances, num_time_steps, num_features = x.shape
    x = np.reshape(x, newshape=(-1, num_features))
    return x 

def scale(x,scaler) :
    scaled_x = scaler.transform(x)
    return scaled_x

def reshape(scaled_x , x) :
    num_instances, num_time_steps, num_features = x.shape
    reshaped_scaled_x =\
    np.reshape(scaled_x, newshape=(num_instances, num_time_steps, num_features))
    return reshaped_scaled_x

# %%
scaler = StandardScaler().fit(flatten(x_train_y0))
x_train_y0_scaled = reshape(scale(flatten(x_train_y0), scaler),x_train_y0)
x_valid_scaled = reshape(scale(flatten(x_valid), scaler),x_valid)
x_valid_y0_scaled = reshape(scale(flatten(x_valid_y0), scaler),x_valid_y0)
x_test_scaled = reshape(scale(flatten(x_test), scaler),x_test)
# %%
timesteps =  x_train_y0_scaled.shape[1] # equal to the lookback
n_features =  x_train_y0_scaled.shape[2] # 59
timesteps , n_features
# %%
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = (
            embedding_dim, 2 * embedding_dim
        )
        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        self.rnn2 = nn.LSTM(
          input_size=self.hidden_dim,
          hidden_size=embedding_dim,
          num_layers=1,
          batch_first=True
        )
    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return  x[:,-1,:]
# %%
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y
# %%
class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
          input_size=input_dim,
          hidden_size=input_dim,
          num_layers=1,
          batch_first=True
        )
        self.rnn2 = nn.LSTM(
          input_size=input_dim,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        self.output_layer = torch.nn.Linear(self.hidden_dim, n_features)
        self.timedist = TimeDistributed(self.output_layer)
        
    def forward(self, x):
        x=x.reshape(-1,1,self.input_dim).repeat(1,self.seq_len,1)       
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        return self.timedist(x)
# %%
class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)#.to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)#.to(device)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
# %%
device = torch.device("cuda")
model = RecurrentAutoencoder(timesteps, n_features, 128)
model = model.to(device)
# %%
class AutoencoderDataset(Dataset): 
    def __init__(self,x):
        self.x = x
    def __len__(self): 
        return len(self.x)
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x[idx,:,:])
        return x
# %%
def train_model(model, train_dataset, val_dataset, n_epochs,batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    print("start!")
    train_dataset_ae = AutoencoderDataset(train_dataset)
    tr_dataloader = DataLoader(train_dataset_ae, batch_size=batch_size, 
                               shuffle=False,num_workers=8)
    val_dataset_ae = AutoencoderDataset(val_dataset)
    va_dataloader = DataLoader(val_dataset_ae, batch_size=len(val_dataset),
                               shuffle=False,num_workers=8)


    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        for batch_idx, batch_x in enumerate(tr_dataloader):
            optimizer.zero_grad()
            batch_x_tensor = batch_x.to(device)
            seq_pred = model(batch_x_tensor)
            loss = criterion(seq_pred, batch_x_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            va_x  =next(va_dataloader.__iter__())
            va_x_tensor = va_x.to(device)
            seq_pred = model(va_x_tensor)
            loss = criterion(seq_pred, va_x_tensor)
            val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    model.load_state_dict(best_model_wts)
    return model.eval(), history
# %%

model, history = train_model(model, x_train_y0_scaled , x_train_y0_scaled , 
                             n_epochs = 500, batch_size=50)
# %%
