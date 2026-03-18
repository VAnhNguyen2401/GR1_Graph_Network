import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv


# ─────────────────────────────────────────
# Định nghĩa cả 2 model (giống lúc train)
# ─────────────────────────────────────────
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


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


# ─────────────────────────────────────────
# Load đồ thị và model
# ─────────────────────────────────────────
device = torch.device('cpu')

ckpt        = torch.load('/content/drive/MyDrive/gnn_models/nslkdd_graph.pt',
                         map_location=device, weights_only=False)
data        = ckpt['data'].to(device)
classes     = ckpt['classes']
num_classes = len(classes)
print(f"   Số node: {data.num_nodes} | Số nhãn: {num_classes}")
print()

# Load GCN
gcn_ckpt = torch.load('/content/drive/MyDrive/gcn_models/gcn_nslkdd_best.pth',
                      map_location=device, weights_only=False)
gcn = GCN(data.num_node_features, 16, num_classes).to(device)
gcn.load_state_dict(gcn_ckpt['model_state_dict'])
gcn.eval()
print(f"   Val Acc: {gcn_ckpt['best_val_acc']:.4f} | Test Acc: {gcn_ckpt['test_acc']:.4f}")

# Load GAT

gat_ckpt = torch.load('/content/drive/MyDrive/gnn_models/gat_nslkdd_best.pth',
                      map_location=device, weights_only=False)
gat = GAT(data.num_node_features, 8, num_classes, heads=8).to(device)
gat.load_state_dict(gat_ckpt['model_state_dict'])
gat.eval()
print(f"   Val Acc: {gat_ckpt['best_val_acc']:.4f} | Test Acc: {gat_ckpt['test_acc']:.4f}")
print()

# ─────────────────────────────────────────
# Dự đoán trên tập TEST
# ─────────────────────────────────────────
with torch.no_grad():
    out_gcn = gcn(data.x, data.edge_index)
    out_gat = gat(data.x, data.edge_index)
    pred_gcn = out_gcn.argmax(dim=-1)
    pred_gat = out_gat.argmax(dim=-1)

test_indices = data.test_mask.nonzero(as_tuple=True)[0]

print("=" * 70)
print(f"{'Node':>6} | {'GCN':^20} | {'GAT':^20} | {'Thực tế':^15} | Đồng ý?")
print("=" * 70)

gcn_correct = gat_correct = 0
for i in test_indices[:20]:
    gcn_pred  = classes[pred_gcn[i].item()]
    gat_pred  = classes[pred_gat[i].item()]
    actual    = classes[data.y[i].item()]
    gcn_ok    = "✅" if pred_gcn[i] == data.y[i] else "❌"
    gat_ok    = "✅" if pred_gat[i] == data.y[i] else "❌"

    if pred_gcn[i] == data.y[i]: gcn_correct += 1
    if pred_gat[i] == data.y[i]: gat_correct += 1

    print(f"{i.item():>6} | {gcn_pred+gcn_ok:^20} | {gat_pred+gat_ok:^20} | {actual:^15}" )

print("=" * 70)
print(f"GCN đúng: {gcn_correct}/20 = {gcn_correct/20*100:.0f}%")
print(f"GAT đúng: {gat_correct}/20 = {gat_correct/20*100:.0f}%")
print()

# ─────────────────────────────────────────
# Dự đoán chi tiết 1 node bất kỳ
# ─────────────────────────────────────────
print(" Dự đoán 1 node bất kỳ:")
node_id = int(input(f"   Nhập node_id (0 - {data.num_nodes-1}): "))

with torch.no_grad():
    prob_gcn = F.softmax(out_gcn[node_id], dim=0)
    prob_gat = F.softmax(out_gat[node_id], dim=0)

actual = classes[data.y[node_id].item()]
print(f"\n📄 Node {node_id} | Thực tế: {actual}")
print()
print(f"{'Loại':<12} | {'GCN':>8} | {'GAT':>8}")
print("-" * 35)
for i, cls in enumerate(classes):
    bar_gcn = "█" * int(prob_gcn[i].item() * 20)
    bar_gat = "█" * int(prob_gat[i].item() * 20)
    mark = " ←" if i == data.y[node_id].item() else ""
    print(f"{cls:<12} | {prob_gcn[i]*100:>6.1f}% | {prob_gat[i]*100:>6.1f}%{mark}")

print()
gcn_final = classes[prob_gcn.argmax().item()]
gat_final = classes[prob_gat.argmax().item()]
print(f"GCN dự đoán: {gcn_final} {'✅' if gcn_final==actual else '❌'}")
print(f"GAT dự đoán: {gat_final} {'✅' if gat_final==actual else '❌'}")
