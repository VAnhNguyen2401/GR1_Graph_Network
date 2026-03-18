import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv

# 1. Dinh nghia model (giong luc train)
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 2. Load do thi + model
device = torch.device('cpu')

print("Dang load do thi...")
graph_ckpt = torch.load(
    '/content/drive/MyDrive/gnn_models/nslkdd_graph.pt',
    map_location=device, weights_only=False
)
data        = graph_ckpt['data'].to(device)
classes     = graph_ckpt['classes']
num_classes = len(classes)

print("Dang load model...")
model_ckpt = torch.load(
    '/content/drive/MyDrive/gnn_models/gat_nslkdd_best.pth',
    map_location=device, weights_only=False
)

model = GAT(
    in_channels     = data.num_node_features,
    hidden_channels = 8,
    out_channels    = num_classes,
    heads           = 8,
).to(device)

model.load_state_dict(model_ckpt['model_state_dict'])
model.eval()

print(f"Load xong!")
print(f"  Best Val Acc luc train: {model_ckpt['best_val_acc']:.4f}")
print(f"  Test Acc luc train:     {model_ckpt['test_acc']:.4f}")
print(f"  Cac nhan: {list(classes)}")
print()

# 3. Du doan toan bo tap TEST
with torch.no_grad():
    out  = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)

test_indices = data.test_mask.nonzero(as_tuple=True)[0]
show         = min(20, len(test_indices))

print("=" * 60)
print(f"{'Node':>6} | {'Du doan':<15} | {'Thuc te':<15} | OK?")
print("=" * 60)

correct = 0
for i in test_indices[:show]:
    predicted  = classes[pred[i].item()]
    actual     = classes[data.y[i].item()]
    is_correct = "OK" if pred[i] == data.y[i] else "SAI"
    if pred[i] == data.y[i]:
        correct += 1
    print(f"{i.item():>6} | {predicted:<15} | {actual:<15} | {is_correct}")

print("=" * 60)
print(f"Dung {correct}/{show} = {correct/show*100:.0f}%")
print()

# 4. Thong ke toan bo tap test
all_pred   = pred[test_indices]
all_actual = data.y[test_indices]
total_acc  = (all_pred == all_actual).sum().item() / len(test_indices)

print(f"Tong node test:  {len(test_indices)}")
print(f"Do chinh xac:    {total_acc*100:.2f}%")
print()
print(f"  {'Loai':<15} | {'Dung':>6} | {'Tong':>6} | {'Acc':>8}")
print("  " + "-" * 42)

for i, cls in enumerate(classes):
    mask  = (all_actual == i)
    total = mask.sum().item()
    if total == 0:
        continue
    right = (all_pred[mask] == i).sum().item()
    acc   = right / total * 100
    print(f"  {cls:<15} | {right:>6} | {total:>6} | {acc:>7.1f}%")

print()

# 5. Du doan 1 node cu the
print("Thu du doan 1 node bat ky:")
node_id = int(input(f"  Nhap node_id (0 - {data.num_nodes-1}): "))

with torch.no_grad():
    out         = model(data.x, data.edge_index)
    pred_single = out[node_id].argmax().item()
    probs       = F.softmax(out[node_id], dim=0)
    confidence  = probs.max().item()

print(f"\nNode {node_id}:")
print(f"  Du doan : {classes[pred_single]} ({confidence*100:.1f}% confidence)")
print(f"  Thuc te : {classes[data.y[node_id].item()]}")
print(f"  Ket qua : {'Dung' if pred_single == data.y[node_id].item() else 'Sai'}")
print()
print(f"Diem so {num_classes} loai:")
for i, (name, prob) in enumerate(zip(classes, probs.tolist())):
    bar  = "#" * int(prob * 30)
    mark = " <--" if i == pred_single else ""
    print(f"  {name:<15} {prob*100:5.1f}% {bar}{mark}")
