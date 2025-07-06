import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
import random

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha,nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.softmax = nn.Softmax(dim=1)
        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))#
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))#
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.softmax = nn.Softmax(dim=1)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        try:
            input = input.float()
        except:
            pass
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class GraphGCNNet(nn.Module): # nfeat, nhid, nclass
    def __init__(self, nfeat, nhid, nclass, adj):
        super(GraphGCNNet, self).__init__()
        self.adj = adj
        self.linear_org = nn.Linear(nfeat, nhid * 2)
        self.conv = GCN(nhid * 2, nhid*3, nhid * 3, dropout=0.3)

        self.linear_3 = nn.Linear(nhid*2, nhid*3)
        self.linear = nn.Linear(nhid*3, nclass)

        self.weight = nn.Parameter(torch.FloatTensor(nhid*3, nhid*3))
        self.lamda = 0.5


    def forward(self, x, adj):
        x = self.linear_org(x)
        x = F.dropout(x, 0.3)

        x_gcn = self.conv(x, self.adj)
        x_gcn = F.dropout(x_gcn, 0.2)

        x = self.linear_3(x) + self.lamda * x_gcn
        x = self.linear(x)

        min_val = torch.min(x)# 找出 x 中的最大值和最小值
        max_val = torch.max(x)

        normalized_x = 2 * (x - min_val) / (max_val - min_val) - 1 # 使用线性变换公式进行归一化


        return normalized_x

class GraphSAGENEW(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGENEW, self).__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels)
        self.conv2 = SAGEConv(2 * out_channels, 2 * out_channels)
        self.conv3 = SAGEConv(2 * out_channels, 2 * out_channels)
        self.conv4 = SAGEConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        return x

from contrast import Contrast

class GenCENet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj):
        """Dense version of GAT."""
        super(GenCENet, self).__init__()
        self.dropout = 0.1
        self.alpha = 0.1
        self.nheads = 4 # 能够被 64整除

        self.gat_layer1 = GATConv(in_channels=nfeat, out_channels=nhid, heads=self.nheads)
        self.gat_layer2 = GATConv(in_channels=nhid*self.nheads, out_channels=nclass, heads=1)

        self.adj = adj
        self.adj = self.adj.coalesce()
        non_zero_indices = torch.nonzero(self.adj.values()).squeeze()
        # 随机打乱索引
        shuffled_indices = torch.randperm(len(non_zero_indices))
        # 计算分割点
        split_point = len(shuffled_indices) // 2

        # 分割索引
        part1_indices = shuffled_indices[:split_point]
        part2_indices = shuffled_indices[split_point:]

        # 根据分割的索引创建两个新的稀疏张量
        part1_indices_original = non_zero_indices[part1_indices]
        part2_indices_original = non_zero_indices[part2_indices]

        self.adj_1 = self.adj.indices()[:, part1_indices_original]
        self.adj_2 = self.adj.indices()[:, part2_indices_original]

        self.adj_1_ = torch.sparse_coo_tensor(
            self.adj.indices()[:, part1_indices_original],
            self.adj.values()[part1_indices_original],
            self.adj.size()
        )

        self.adj_2_ = torch.sparse_coo_tensor(
            self.adj.indices()[:, part2_indices_original],
            self.adj.values()[part2_indices_original],
            self.adj.size()
        )

        self.contrast = Contrast(nclass, 0.3, 0.3)
        self.softmax = nn.Softmax(dim=1)

        self.linear_1 = nn.Linear(nclass, nclass*3)
        self.linear_2 = nn.Linear(nclass*3, nclass)

    def forward(self, x, adj):
        x_2 = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat_layer1(x, self.adj_1.cuda())

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat_layer2(x, self.adj_1)
        min_val, _ = torch.min(x, dim=1, keepdim=True)
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        normalized_x = 4 * (x - min_val) / (max_val - min_val) - 2 # 使用线性变换公式进行归一化
        normalized_x = self.linear_2(self.linear_1(normalized_x))

        x_2 = self.gat_layer1(x_2, self.adj_2)
        # torch.cat([att(x_2, self.adj_2) for att in self.attentions], dim=1)  # 在这进图注意层
        x_2 = F.dropout(x_2, self.dropout, training=self.training)
        x_2 = self.gat_layer2(x_2, self.adj_2)

        min_val_2, _ = torch.min(x_2, dim=1, keepdim=True)
        max_val_2, _ = torch.max(x_2, dim=1, keepdim=True)

        normalized_x_2 = 4 * (x_2 - min_val_2) / (max_val_2 - min_val_2) - 2 # 使用线性变换公式进行归一化
        normalized_x_2 = self.linear_2(self.linear_1(normalized_x_2))

        total_loss = 0
        batch_size = 3000
        num_batches = len(normalized_x) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_x = normalized_x[start_idx:end_idx]
            batch_x_2 = normalized_x_2[start_idx:end_idx]
            batch_mask = torch.eye(len(batch_x_2)).cuda()

            batch_loss = self.contrast(F.relu(batch_x), F.relu(batch_x_2), batch_mask)
            total_loss += batch_loss


        return 0.5 * (normalized_x+normalized_x_2), total_loss



class LightGCN(nn.Module):
    def __init__(self, num_nodes, nclass, num_layers, nfeat, dropout):
        super(LightGCN, self).__init__()
        self.num_nodes = num_nodes
        self.nclass = nclass
        self.num_layers = num_layers
        self.nfeat = nfeat
        self.dropout = dropout
        self.linear = nn.Linear(nfeat, nclass)

        self.embedding_nodes = nn.Embedding(
            num_embeddings=self.num_nodes, embedding_dim=self.nfeat)

    def forward(self, x, adj_mat):
        all_embeddings = x
        all_embed = []
        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(adj_mat, all_embeddings)
            all_embed.append(all_embeddings)

        all_embeds = torch.stack(all_embed, 1)
        all_embeds = self.linear(torch.mean(all_embeds, dim=1))

        return F.dropout(all_embeds, self.dropout, training=self.training)



class GCN2Layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN2Layer, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        # self.conv3 = GCNConv(2 * out_channels, 2 * out_channels)
        # self.conv4 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))
        # x = self.conv4(x, edge_index)
        return x

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATModel, self).__init__()
        # 定义两层GAT模型，首层8个头，第二层1个头
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=4, concat=False, dropout=0.1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x








