import os

import torch
import numpy as np
import scipy.sparse as sp


def load_data(dataset="cora"):

    print("Loading {} dataset ...".format(dataset), end="", flush=True)

    path = "./data/" + dataset + "/"
    pt_path = path + dataset + ".pt"

    if os.path.exists(pt_path):
        return torch.load(pt_path)

    if dataset == "cora" or dataset == "citeseer":
        sparse = (dataset == "citeseer")

        idx_features_labels = np.genfromtxt(path + dataset + ".content", dtype=np.dtype(str))

        features = normalize_features(sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32), sparse=sparse)
        labels = encode_labels(idx_features_labels[:, -1])

        idx_dict = {j: i for i, j in enumerate(idx_features_labels[:, 0])}

        edges_raw = np.genfromtxt(path + dataset + ".cites", dtype=np.dtype(str))
        edges_converted = list(map(lambda x: idx_dict.get(x, -1), edges_raw.flatten()))
        edges = torch.LongTensor(edges_converted).reshape(edges_raw.shape)
        edges = edges[~(edges < 0).any(axis=1)]

        adj = create_adj(edges, (labels.shape[0],) * 2)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]), sparse=sparse)

        if dataset == "cora":
            idx_train = torch.LongTensor(range(140))
            idx_val = torch.LongTensor(range(200, 500))
            idx_test = torch.LongTensor(range(500, 1500))

            # # Data from torch_geometric
            # idx_train = torch.LongTensor(range(0, 140))
            # idx_val = torch.LongTensor(range(140, 640))
            # idx_test = torch.LongTensor(range(1708, 2708))
        else:
            idx_train = torch.LongTensor(range(120))
            idx_val = torch.LongTensor(range(200, 500))
            idx_test = torch.LongTensor(range(500, 1500))

            # # Data from torch_geometric
            # idx_train = torch.LongTensor(range(120))
            # idx_val = torch.LongTensor(range(120, 620))
            # idx_test = torch.LongTensor(range(2312, 3312))
    else:
        raise ValueError("Invalid dataset '{}'".format(dataset))

    torch.save((adj, features, labels, idx_train, idx_val, idx_test), pt_path)

    return adj, features, labels, idx_train, idx_val, idx_test


def accuracy(output, labels):
    preds = output.max(1)[1]
    correct = preds.eq(labels).sum().item()
    return correct / len(labels)


def create_adj(edges, shape):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), edges.T), shape=shape, dtype=np.float32)
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


def normalize_adj(mx, sparse=False):  # A_hat = DAD
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    mx_to = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    if sparse:
        mx_to = mx_to.tocoo()
        indices = torch.LongTensor([mx_to.row.tolist(), mx_to.col.tolist()])
        values = torch.Tensor(mx_to.data)
        return torch.sparse.FloatTensor(indices, values)
    else:
        return torch.FloatTensor(mx_to.todense())


def normalize_features(mx, sparse=False):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx_to = r_mat_inv.dot(mx)

    if sparse:
        mx_to = mx_to.tocoo()
        indices = torch.LongTensor([mx_to.row.tolist(), mx_to.col.tolist()])
        values = torch.Tensor(mx_to.data)
        return torch.sparse.FloatTensor(indices, values)
    else:
        return torch.FloatTensor(mx_to.todense())


def encode_labels(labels):
    classes = {c: i for i, c in enumerate(np.unique(labels))}
    return torch.LongTensor(list(map(classes.get, labels)))
