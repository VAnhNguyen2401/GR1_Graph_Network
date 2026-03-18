import argparse
import os.path as osp
import os
import time

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.logging import log
from torch_geometric.nn import GATConv


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=8)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    return parser


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels,
                             heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def main(args=None):
    parser = build_parser()
    args   = parser.parse_args(args=args)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch_geometric.is_xpu_available():
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')
    print(f"Dung device: {device}")

    # Load do thi NSL-KDD
    print("Dang load do thi NSL-KDD...")
    ckpt        = torch.load('/content/drive/MyDrive/gnn_models/nslkdd_graph.pt',
                             map_location=device)
    data        = ckpt['data'].to(device)
    classes     = ckpt['classes']
    num_classes = len(classes)

    print(f"  So node:     {data.num_nodes}")
    print(f"  So canh:     {data.num_edges}")
    print(f"  Feature dim: {data.num_node_features}")
    print(f"  So nhan:     {num_classes} → {list(classes)}")
    print()

    model = GAT(
        in_channels     = data.num_node_features,
        hidden_channels = args.hidden_channels,
        out_channels    = num_classes,
        heads           = args.heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=5e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def test():
        model.eval()
        pred = model(data.x, data.edge_index).argmax(dim=-1)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        return accs

    times        = []
    best_val_acc = test_acc = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss  = train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc     = tmp_test_acc
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
        times.append(time.time() - start)

    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

    # Luu model
    save_dir  = '/content/drive/MyDrive/gnn_models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, 'gat_nslkdd_best.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_val_acc'    : best_val_acc,
        'test_acc'        : test_acc,
        'args'            : vars(args),
        'classes'         : classes,
    }, save_path)
    print(f"Da luu model tai: {save_path}")
    print(f"  Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == '__main__':
    main()
