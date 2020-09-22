import torch
from torch import nn
from torch import sparse
from torch.nn import init
from torch.nn import functional as F


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GraphConvolutionLayer, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        init.xavier_uniform_(self.weight, gain=init.calculate_gain("relu"))

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.mm(adj, x)
        x = torch.mm(x, self.weight)
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, nheads, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_features, nheads * out_features))
        init.xavier_uniform_(self.weight, gain=init.calculate_gain("leaky_relu"))

        self.linear_i = nn.Parameter(torch.Tensor(1, nheads, out_features))
        init.xavier_uniform_(self.linear_i, gain=init.calculate_gain("leaky_relu"))

        self.linear_j = nn.Parameter(torch.Tensor(1, nheads, out_features))
        init.xavier_uniform_(self.linear_j, gain=init.calculate_gain("leaky_relu"))

        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads
        self.concat = concat

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        wh = torch.mm(x, self.weight).view(-1, self.nheads, self.out_features)

        awh_i = (wh * self.linear_i).sum(dim=2).unsqueeze(dim=1)
        awh_j = (wh * self.linear_j).sum(dim=2).unsqueeze(dim=0)

        mask = (-1e10 * (1 - adj)).unsqueeze(dim=2)
        e = F.leaky_relu(awh_i + awh_j, negative_slope=self.alpha) + mask
        a = F.softmax(e, dim=1).unsqueeze(dim=3)
        wh = wh.unsqueeze(dim=0)

        x = (a * wh).sum(dim=1)

        if self.concat:
            return x.flatten(start_dim=1)
        else:
            return x.mean(dim=1)


class SparseGraphConvolutionLayer(GraphConvolutionLayer):
    def forward(self, x, adj):
        if x.is_sparse:
            x = sparse.mm(x, self.weight)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.mm(x, self.weight)
        x = sparse.mm(adj, x)
        return x


class SparseGraphAttentionLayer(GraphAttentionLayer):
    def forward(self, x, adj):
        if x.is_sparse:
            wh = sparse.mm(x, self.weight).view(-1, self.nheads, self.out_features)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            wh = torch.mm(x, self.weight).view(-1, self.nheads, self.out_features)

        awh_i = (wh * self.linear_i).sum(dim=2)
        awh_j = (wh * self.linear_j).sum(dim=2)

        idx_i, idx_j = adj._indices()

        e_values = F.leaky_relu(awh_i[idx_i] + awh_j[idx_j], negative_slope=self.alpha)

        e = sparse.FloatTensor(adj._indices(), e_values)
        a = sparse.softmax(e.cpu(), dim=1).to(e.device)

        # Choose memory / speed tradeoff
        # keep_sparse = True  : Loop through sparse tensor (Low memory usage / Slow)
        # keep_sparse = False : Convert sparse tensor to dense tensor (High memory usage / Fast)
        # Both methods return almost identical results
        keep_sparse = False

        if keep_sparse:
            x = torch.cat([
                (a[i]._values().unsqueeze(dim=2) * wh[a[i]._indices()[0]]).sum(dim=0, keepdim=True)
                for i in range(x.shape[0])
            ], dim=0)
        else:
            a = a.to_dense().unsqueeze(dim=3)
            wh = wh.unsqueeze(dim=0)
            x = (a * wh).sum(dim=1)

        if self.concat:
            return x.flatten(start_dim=1)
        else:
            return x.mean(dim=1)
