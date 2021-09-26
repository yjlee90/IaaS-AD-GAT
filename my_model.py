# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import easydict
from tqdm import tqdm
# from my_autoencoder_graph import *
# from my_modules import *
from my_dataset import SlidingWindowDataset

# %%
# model definition
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class MyGATNet(nn.Module) :
    def __init__(self, args):
        super(MyGATNet, self).__init__()
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels

        self.conv1 = GATConv(self.in_channels, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(
            8 * 8, self.out_channels, heads=2, concat=False, dropout=0.6)

    def forward(self, x, edge_index) :
        for t in range(len(x)) :




        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)

# %%
# training phase

# hyperparameter setting
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

 # %%
model = MyModel(args)
model = model.to(args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
epochs = tqdm(range(args.max_iter//len(train_loader)+1))

# %%
for epoch in epochs:
    model.train()
    optimizer.zero_grad()
    train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="training")

    for i, batch_data in train_iterator:
        past_data = batch_data
        future_data = batch_data
        batch_size = past_data.size(0)
        example_size = past_data.size(1)
        past_data = past_data.view(batch_size, example_size, -1).float().to(args.device)
        future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)


        model(past_data)


# %%

# %%
