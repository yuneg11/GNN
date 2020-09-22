import random
import argparse
from time import time

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

from model import GCN, GAT, SpGCN, SpGAT
from utils import accuracy, load_data


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.epoch = 0

        self.prev_loss = 1e+9
        self.prev_acc = 0.
        self.prev_epoch = 0

    def update(self, loss, acc):
        self.epoch += 1

        if loss < self.prev_loss:
            self.prev_epoch = self.epoch
            self.prev_loss = loss

        if acc > self.prev_acc:
            self.prev_epoch = self.epoch
            self.prev_acc = acc

        return self.prev_epoch + self.patience < self.epoch


if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["gcn", "gat"], help="GCN or GAT")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disables CUDA training.")
    parser.add_argument("--fastmode", action="store_true", default=False, help="Validate during training pass.")
    parser.add_argument("--sparse", action="store_true", default=False, help="GAT with sparse version or not.")
    parser.add_argument("--seed", type=int, default=72, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train.")
    parser.add_argument("--save_every", type=int, default=10, help="Save every n epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay (L2 loss on parameters).")
    parser.add_argument("--hidden", type=int, default=8, help="Number of hidden units.")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of head attentions.")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate (1 - keep probability).")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu.")
    parser.add_argument("--patience", type=int, default=10, help="patience")
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "citeseer"], help="Dataset to train.")
    parser.add_argument("-d", "--device")

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # device = torch.device("cuda:2" if args.cuda else "cpu")
    device = torch.device(args.device if args.cuda else "cpu")

    # Load dataset
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    nfeatures = features.shape[1]
    nclass = labels.max() + 1

    # Load model
    if args.model == "gcn":
        if args.sparse:
            model = SpGCN(nfeatures, args.hidden, nclass, args.dropout)
        else:
            model = GCN(nfeatures, args.hidden, nclass, args.dropout)
    elif args.model == "gat":
        if args.sparse:
            model = SpGAT(nfeatures, args.hidden, nclass, args.dropout, args.alpha, args.n_heads)
        else:
            model = GAT(nfeatures, args.hidden, nclass, args.dropout, args.alpha, args.n_heads)
    else:
        raise ValueError("Invalid model '{}'".format(args.model))

    # Move to device
    model.to(device)
    adj, features, labels = adj.to(device), features.to(device), labels.to(device)

    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    early_stopping = EarlyStopping(args.patience)

    start_time = time()

    for epoch in range(1, args.epochs + 1):
        epoch_time = time()

        # Train
        model.train()
        optimizer.zero_grad()

        out = model(features, adj)
        train_acc = accuracy(out[idx_train], labels[idx_train])

        loss = F.cross_entropy(out[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        out = model(features, adj)

        val_acc = accuracy(out[idx_val], labels[idx_val])
        test_acc = accuracy(out[idx_test], labels[idx_test])

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), "model/" + model.__class__.__name__ + "-" + args.dataset + "-" + str(epoch) + ".pt")

        print("\rEpoch {:3d}: Loss {:.3f} / Train acc. {:4.1f}% / Val acc. {:4.1f}% / Test acc. {:4.1f}% / {:.3f}s".format(
            epoch, loss.item(), train_acc * 100., val_acc * 100., test_acc * 100., time() - epoch_time
        ), end=("" if epoch % args.save_every else "\n"), flush=True)

        epoch_time = time()

        if early_stopping.update(loss, val_acc):
            break

    print("\nTraining time: {:.3f}s".format(time() - start_time))
