# %%
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from dgl import DGLGraph
import dgl

from dgl.data import CitationGraphDataset
citeseer = CitationGraphDataset('citeseer')

# %%
# dir(citeseer)

# networkx 라이브러리의 오브젝트인 그래프가 citeseer.graph 로 들어가 있다.
# draw 함수를 통해 아래와 같은 끔찍한 이미지를 만들어 낼 수 있다.
import networkx as nx

nx_G = citeseer.graph.to_undirected()
pos = nx.kamada_kawai_layout(nx_G)
display(nx.draw(nx_G, pos, with_labels=False, node_size = 0.01, node_color='#00b4d9'))


# %%
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        
        # Expression 3
        # F-Dimension의 피쳐 스페이스가 single fc-layer 지나며 F'-Dimension으로 임베딩 
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # i노드의 F' + j노드의 F' 길이의 벡터를 합쳐서 Attention Coefficient를 리턴 	
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

        
    # Expression 3에서 어텐션으로 넘어온 값을 Leaky Relu 적용하는 Layer
	# src는 source vertex, dst는 destination vertex의 약자	
    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    
    # dgl에서는 모든 노드에 함수를 병렬 적용 할 수 있는 update_all 이라는 api를 제공한다.
    # 해당 api 사용을 위해 텐서를 흘려보내는 역할을 한다고 한다.
	# 구체적인 update_all의 알고리즘은 잘 모르겠으니 그냥 input 함수라고 생각하자.
    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}


    # update_all에서는 흘려보내진 텐서를 각 노드의 mailbox라는 오브젝트에 저장하나 보다.
    # 각 노드에는 여러 이웃이 있으니 mailbox에는 여러개의 attention coefficient가 있다.
    # Expression 4에서 softmax 계수를 가중하여 element wise하게 합한다.  
    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    
    # (1) fc layer를 통해 피쳐를 임베딩
    # (2) 그레프에 임베딩 된 벡터를 저장
    # (3) apply_edges api를 모든 엣지에 적용하여 i - j 간의 attention coefficeint를 계산
    # (4) 그래프에 저장된 z와e를 텐서로 reduce_func에 전달하여 새로운 h' 를 얻는다.
    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')
# %%
class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))
# %%
class GAT(nn.Module):
    
    # 두 Layer의 인풋과 아웃풋이 다른 것을 볼 수 있다
    # 원래 노드의 feature 개수가 F개라고 했을 때, layer를 한 번 지나며 F'개로 임베딩했다.
    # 이것을 num_heads(attention 개수) 만큼 multi-head하게 보아 K*F' 길이로 cat했다.
    # 두 번째 layer에서는 K를 1로 설정하여 single-head attention을 적용했다.  
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h
# %%
def load_citeseer_data():
    data = citeseer
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.BoolTensor(data.train_mask)
    g = data.graph
    
    # add self loop
    # GAT는 i <-> i의 self-attention도 종합하기 때문에 해당 정보를 edge에 추가해준다
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask
# %%
g, features, labels, mask = load_citeseer_data()

# create the model, 2 heads, each head has hidden size 8
net = GAT(g,
          in_dim=features.size()[1],
          hidden_dim=8,
          out_dim=6,
          num_heads=2)

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# main loop
dur = []
train_loss_arr = []
for epoch in range(1000):
    if epoch >= 3:
        t0 = time.time()

    logits = net(features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss_arr.append(loss.item())

    if epoch >= 3:
        dur.append(time.time() - t0)
        
    if epoch % 100 == 0:
      print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
          epoch, loss.item(), np.mean(dur)))

    
# %%
import matplotlib.pyplot as plt
plt.plot(train_loss_arr)
display(plt.show())
# %%
