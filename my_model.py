import torch
import torch.nn as nn
from my_modules import *

class MyModel(nn.Module) :
    def __init__(self, n_features,window_size,dropout,alpha):
        super(MyModel, self).__init__()

        self.conv = ConvLayer(
            n_features = n_features
        )

        self.gat_layer = GraphAttentionLayer(
            n_features = n_features,
            window_size = window_size,
            dropout = dropout,
            alpha = alpha
        )

    def forward(self, x) :
        h = self.conv(x)
        result = self.gat_layer(h)

        return result