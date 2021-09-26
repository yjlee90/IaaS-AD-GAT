# %%
import torch
from dgl import DGLGraph
import dgl

# %%
import numpy as np
def build_dummie_cloud_graph():

    '''
    6, 7, 8    9,10,11   12,13,14   15,16,17   18,19,20
    --------   --------  --------  --------    --------
       1          2         3          4           5
    
    6,7,9,12,15,16
    8,10,11,12,13,14,17,18,19,20
    '''
    
    src = np.array([1,2,1,1,1,2,2,2,3,3,4,5,5,7],)    
    dst = np.array([2,1,3,4,5,6,7,8,4,6,6,7,8,8])

    u = np.concatenate([src,dst])
    v = np.concatenate([dst,src])
    return dgl.DGLGraph((u,v))


#%%
import networkx as nx
# Since the actual graph is undirected, we convert it for visualization
# purpose.
G = build_dummie_cloud_graph()
G = dgl.add_self_loop(G)
nx_G = G.to_networkx().to_undirected()
# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
# %%

import torch
import torch.nn as nn
import torch.nn.functional as F  

embed = nn.Embedding(9, 20)
G.ndata['feat'] = embed.weight

# print out node 2's input feature
print(G.ndata['feat'][2])

# print out node 10 and 11's input features
print(G.ndata['feat'][[3, 5]])

# %%

# %%

import itertools

optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
all_logits = []
for epoch in range(50):
    logits = net(G, inputs)
    # we save the logits for visualization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

# %%

import matplotlib.animation as animation
import matplotlib.pyplot as plt

def draw(i):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(9):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
            with_labels=True, node_size=300, ax=ax)

fig = plt.figure(dpi=150)
fig.clf()
ax = fig.subplots()
draw(0)  # draw the prediction of the first epoch
plt.close()

# %%
ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)

# %%


class GCN_layer(nn.Module) :
    def __init__(self, in_features, out_features, A):
        super(GCN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A = A
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, X) :
        return self.fc(torch.spmm(self.A, X))
    
    
        
# %%


class GATLayer(nn.Module) :
    def __init__(self, g, in_dim, out_dim) :
        super(GATLayer, self).__init__()
        self.g = g 
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges) :
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e' : F.leaky_relu(a)}
    
   
    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    # update_all에서는 흘려보내진 텐서를 각 노드의 mailbox라는 오브젝트에 저장하나 보다.
    # 각 노드에는 여러 이웃이 있으니 mailbox에는 여러개의 attention coefficient가 있다.
    # Expression 4에서 softmax 계수를 가중하여 element wise하게 합한다.  
    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h} 

    def forward(self, h):
       z = self.fc(h)
       self.g.ndata['z'] = z
       self.g.apply_edges(self.edge_attention)
       self.g.update_all(self.message_func, self.reduce_func)
       return self.g.ndata.pop('h')

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


from dgl.data import CitationGraphDataset
from dgl import DGLGraph
citeseer = CitationGraphDataset('citeseer')


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

# %%
