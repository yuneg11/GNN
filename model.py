import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import GraphConvolutionLayer, GraphAttentionLayer, SparseGraphConvolutionLayer, SparseGraphAttentionLayer


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.layer1 = GraphConvolutionLayer(nfeat, nhid, dropout)
        self.relu = nn.ReLU()
        self.layer2 = GraphConvolutionLayer(nhid, nclass, dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.relu(x)
        x = self.layer2(x, adj)
        if not self.training:
            x = self.softmax(x)
        return x


# TODO step 2.
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        pass

    def forward(self, x, adj):
        pass


# TODO step 3.
class SpGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SpGCN, self).__init__()
        pass

    def forward(self, x, adj):
        pass


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(SpGAT, self).__init__()
        pass

    def forward(self, x, adj):
        pass
