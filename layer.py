import torch
from torch import nn
from torch.nn import functional as F


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GraphConvolutionLayer, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

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
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("leaky_relu"))

        self.linear_i = nn.Parameter(torch.Tensor(1, nheads, out_features))
        nn.init.xavier_uniform_(self.linear_i, gain=nn.init.calculate_gain("leaky_relu"))

        self.linear_j = nn.Parameter(torch.Tensor(1, nheads, out_features))
        nn.init.xavier_uniform_(self.linear_j, gain=nn.init.calculate_gain("leaky_relu"))

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


# TODO step 3.
class SparsemmFunction(torch.autograd.Function):
    """ for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


class Sparsemm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SparsemmFunction.apply(indices, values, shape, b)


class SparseGraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(SparseGraphConvolutionLayer, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight = nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

        self.dropout = dropout

    def forward(self, x, adj):
        if x.is_sparse:
            x = torch.sparse.mm(x, self.weight)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.mm(x, self.weight)
        x = torch.sparse.mm(adj, x)
        return x


class SparseGraphAttentionLayer(nn.Module):
    """multihead attention """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SparseGraphAttentionLayer, self).__init__()
        pass

    def forward(self, input, adj):
        pass
