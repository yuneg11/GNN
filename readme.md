# GCN and GAT

Cora dataset -> GCN or GAT  \
CiteSeer dataset -> SpGCN or SpGAT

## GCN
```bash
python3 train.py gcn --hidden 64
```
## GAT
```bash
python3 train.py gat
```

## SpGCN
```bash
python3 train.py gcn --hidden 64 --dataset citeseer
```

## SpGAT
```bash
python3 train.py gcn --dataset citeseer
```
