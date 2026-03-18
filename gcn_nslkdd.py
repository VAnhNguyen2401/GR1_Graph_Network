import argparse
import os.path as osp
import os
import time

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.logging import log
from torch_geometric.nn import GCNConv  # ← đổi sang GCNConv


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=16)  # GCN dùng 16
    parser.add_argument('--lr', type=float, default=0.01)           # GCN dùng 0.01
    parser.add_argument('--epochs', type=int, default=200)
    return parser


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)  # GCN dropout 0.5
        x = self.conv1(x, edge_index).relu()              # GCN dùng relu
        x = F.dropout(x, p=0.5, training=self.training)
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
                             map_location=device, weights_only=False)
    data        = ckpt['data'].to(device)
    classes     = ckpt['classes']
    num_classes = len(classes)

    print(f"  So node:     {data.num_nodes}")
    print(f"  So canh:     {data.num_edges}")
    print(f"  Feature dim: {data.num_node_features}")
    print(f"  So nhan:     {num_classes} → {list(classes)}")
    print()

    # GCN không có heads nên đơn giản hơn GAT
    model = GCN(
        in_channels     = data.num_node_features,
        hidden_channels = args.hidden_channels,
        out_channels    = num_classes,
    ).to(device)

    # GCN dùng weight_decay khác nhau cho 2 layer
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0),
    ], lr=args.lr)

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
    save_dir  = '/content/drive/MyDrive/gcn_models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, 'gcn_nslkdd_best.pth')
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
