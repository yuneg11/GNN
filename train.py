import os
import random
import argparse
from time import time

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import optim
from torch.nn import functional as F

from model import GCN, GAT, SpGCN, SpGAT
from utils import accuracy, load_data


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.epoch = 0

        self.best_loss = 1e+9
        self.best_acc = 0.
        self.best_epoch = 0

    def update(self, loss, acc):
        self.epoch += 1

        if loss < self.best_loss:
            self.best_epoch = self.epoch
            self.best_loss = loss

        if acc > self.best_acc:
            self.best_epoch = self.epoch
            self.best_acc = acc

        return self.best_epoch + self.patience < self.epoch


class History:
    def __init__(self):
        self.epochs = []
        self.train_accs = []
        self.val_accs = []
        self.test_accs = []

    def __getitem__(self, epoch):
        return {
            "train": self.train_accs[epoch],
            "val": self.val_accs[epoch],
            "test": self.test_accs[epoch],
        }

    def update(self, epoch, train_acc, val_acc, test_acc):
        self.epochs.append(epoch)
        self.train_accs.append(train_acc * 100.)
        self.val_accs.append(val_acc * 100.)
        self.test_accs.append(test_acc * 100.)

    def save_graph(self, filename, title):
        plt.title(title, fontsize=14)
        plt.plot(self.epochs, self.train_accs, label="Train Acc.")
        plt.plot(self.epochs, self.val_accs, label="Val Acc.")
        plt.plot(self.epochs, self.test_accs, label="Test Acc.")
        plt.legend(loc="lower right")
        plt.xlabel("Epochs")
        plt.xlim(0, len(self.epochs))
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.grid()
        plt.tight_layout()
        plt.savefig(filename)


if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=["gcn", "gat"], help="GCN or GAT")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disable CUDA training")
    parser.add_argument("--seed", type=int, default=72, help="Random seed")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train")
    parser.add_argument("--save_every", type=int, default=10, help="Save every n epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay (L2 loss on parameters)")
    parser.add_argument("--hidden", type=int, default=8, help="Number of hidden units")
    parser.add_argument("--nheads", type=int, default=8, help="Number of head attentions")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate (1 - keep probability)")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "citeseer"], help="Dataset to train")
    parser.add_argument("--device", type=str, default="cuda", help="Desired device to train")
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if args.cuda else "cpu")

    # Load dataset
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    nfeatures = features.shape[1]
    nclass = len(labels.unique())
    sparse = (args.dataset == "citeseer")

    # Load model
    if args.model == "gcn":
        if sparse:
            model = SpGCN(nfeatures, args.hidden, nclass, args.dropout)
        else:
            model = GCN(nfeatures, args.hidden, nclass, args.dropout)
    elif args.model == "gat":
        if sparse:
            model = SpGAT(nfeatures, args.hidden, nclass, args.dropout, args.alpha, args.nheads)
        else:
            model = GAT(nfeatures, args.hidden, nclass, args.dropout, args.alpha, args.nheads)
    else:
        raise ValueError("Invalid model '{}'".format(args.model))

    # Move to device
    model.to(device)
    adj, features, labels = adj.to(device), features.to(device), labels.to(device)

    # Prepare training
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(args.patience)
    history = History()

    start_time = time()
    save_file_name_format = "models/" + model.__class__.__name__ + "-" + args.dataset + "-{}.pt"
    graph_file_name = model.__class__.__name__ + "-" + args.dataset + ".png"
    os.makedirs("./models", exist_ok=True)

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

        print("\rEpoch {:3d}: Loss {:.3f} / Train {:4.1f}% / Val {:4.1f}% / Test {:4.1f}% / Time {:.3f}s".format(
            epoch, loss.item(), train_acc * 100., val_acc * 100., test_acc * 100., time() - epoch_time
        ), end=("" if epoch % args.save_every else "\n"), flush=True)

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), save_file_name_format.format(epoch))

        history.update(epoch, train_acc, val_acc, test_acc)

        if early_stopping.update(loss, val_acc):
            break

    print("\nTraining time: {:.3f}s".format(time() - start_time))

    best_epoch = early_stopping.best_epoch
    print("Best model: Epoch {} / Test Accuracy {:.2f}%".format(best_epoch, history[best_epoch]["test"]))
    history.save_graph(graph_file_name, "{} {}".format(model.__class__.__name__, args.dataset))
