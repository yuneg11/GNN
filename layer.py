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
        n = x.shape[0]

        x = F.dropout(x, p=self.dropout, training=self.training)
        wh = torch.mm(x, self.weight).view(n, self.nheads, self.out_features)

        if self.concat:
            out_features = self.nheads * self.out_features
        else:
            out_features = self.out_features

        awh_i = (wh * self.linear_i).sum(dim=2)
        awh_j = (wh * self.linear_j).sum(dim=2)

        hp = torch.Tensor(n, out_features).to(x.device)

        mask = (1 - adj) * -1e10

        for i in range(n):
            e = F.leaky_relu(awh_i[i] + awh_j, negative_slope=self.alpha)
            e += mask[i].unsqueeze(dim=1)

            a = F.softmax(e, dim=0).unsqueeze(dim=2)

            out = (a * wh).sum(dim=0)

            if self.concat:
                hp[i] = out.flatten()
            else:
                hp[i] = out.mean(dim=0)

        return hp


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
        pass

    def forward(self, input, adj):
        pass


class SparseGraphAttentionLayer(nn.Module):
    """multihead attention """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SparseGraphAttentionLayer, self).__init__()
        pass

    def forward(self, input, adj):
        pass
