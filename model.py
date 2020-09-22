from torch import nn

from layer import (
    GraphConvolutionLayer,
    GraphAttentionLayer,
    SparseGraphConvolutionLayer,
    SparseGraphAttentionLayer,
)


class GCNBase(nn.Module):
    def __init__(self, ConvolutionLayer, nfeat, nhid, nclass, dropout):
        super(GCNBase, self).__init__()
        self.layer1 = ConvolutionLayer(nfeat, nhid, dropout)
        self.relu = nn.ReLU()
        self.layer2 = ConvolutionLayer(nhid, nclass, dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.relu(x)
        x = self.layer2(x, adj)
        if not self.training:
            x = self.softmax(x)
        return x


class GATBase(nn.Module):
    def __init__(self, AttentionLayer, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GATBase, self).__init__()
        self.layer1 = AttentionLayer(nfeat, nhid, dropout, alpha, nheads, concat=True)
        self.elu = nn.ELU()
        self.layer2 = AttentionLayer(nhid * nheads, nclass, dropout, alpha, nheads, concat=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.elu(x)
        x = self.layer2(x, adj)
        if not self.training:
            x = self.softmax(x)
        return x


class GCN(GCNBase):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__(
            GraphConvolutionLayer,
            nfeat, nhid, nclass, dropout
        )


class GAT(GATBase):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__(
            GraphAttentionLayer,
            nfeat, nhid, nclass, dropout, alpha, nheads
        )

    def forward(self, x, adj):
        adj = (adj > 0).float()  # Denormalize
        return super().forward(x, adj)


class SpGCN(GCNBase):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SpGCN, self).__init__(
            SparseGraphConvolutionLayer,
            nfeat, nhid, nclass, dropout
        )


class SpGAT(GATBase):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(SpGAT, self).__init__(
            SparseGraphAttentionLayer,
            nfeat, nhid, nclass, dropout, alpha, nheads
        )
