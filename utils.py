import os

import torch
import numpy as np
import scipy.sparse as sp


def load_data(dataset="cora"):

    print("Loading {} dataset ...".format(dataset), end="", flush=True)

    path = "./data/" + dataset

    if os.path.exists(path + ".pt"):
        return torch.load(path + ".pt")

    if dataset == "cora":
        idx_features_labels = np.genfromtxt(path + "/cora.content", dtype=np.dtype(str))

        features = normalize_features(sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32), sparse=False)
        labels = encode_labels(idx_features_labels[:, -1])

        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}

        edges_unordered = np.genfromtxt(path + "/cora.cites", dtype=np.int32)
        edges = torch.LongTensor(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)

        adj = create_adj(edges)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]), sparse=True)

        idx_train = torch.LongTensor(range(140))
        idx_val = torch.LongTensor(range(200, 500))
        idx_test = torch.LongTensor(range(500, 1500))
    elif dataset == "citeseer":
        # TODO step 3.
        pass
    else:
        raise ValueError("Invalid dataset '{}'".format(dataset))

    torch.save((adj, features, labels, idx_train, idx_val, idx_test), path + ".pt")

    return adj, features, labels, idx_train, idx_val, idx_test


def accuracy(output, labels):
    preds = output.max(1)[1]
    correct = preds.eq(labels).sum().item()
    return correct / len(labels)


def create_adj(edges):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), edges.T), shape=(labels.shape[0],)*2, dtype=np.float32)
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
