# %%
import numpy as np
import pandas as pd
import pickle
import os
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, dataloader, dataset
import torch
from my_dataset import *
from my_autoencoder_graph import *

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


import easydict
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
model = model.to(args.device)
# optimizer 설정
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
epochs = tqdm(range(args.max_iter//len(train_loader)+1))

writer = SummaryWriter(log_dir=f'./runs/my-lstm-AE-ex2-b{args.batch_size}-h{args.hidden_size}', comment="데이터셋 가상 anomaly 제거")

count = 0
running_loss  = 0

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

        running_loss += loss.item()
        if i % 100 == 99:
            writer.add_scalar('training_loss',
                            running_loss/ 100,
                            epoch * len(train_loader) + i )


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

writer.close()

del train_dataset
del test_dataset
torch.cuda.empty_cache()

# %%