import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

columns = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty'
]

# BUOC 1: Doc file
print("Dang doc NSL-KDD...")
train_df = pd.read_csv('/content/drive/MyDrive/Vanh_GR1/KDDTrain+.txt', header=None, names=columns)
test_df  = pd.read_csv('/content/drive/MyDrive/Vanh_GR1/KDDTest+.txt',  header=None, names=columns)
df = pd.concat([train_df, test_df], ignore_index=True)
df = df.drop(columns=['difficulty'])
print(f"  Tong so dong: {len(df)}")

# BUOC 2: Gop nhan thanh 5 loai chinh
print("Dang gop nhan...")
attack_map = {
    'normal': 'NORMAL',
    'neptune':'DoS','back':'DoS','land':'DoS','pod':'DoS',
    'smurf':'DoS','teardrop':'DoS','mailbomb':'DoS','apache2':'DoS',
    'processtable':'DoS','udpstorm':'DoS',
    'ipsweep':'Probe','nmap':'Probe','portsweep':'Probe',
    'satan':'Probe','mscan':'Probe','saint':'Probe',
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L',
    'multihop':'R2L','phf':'R2L','spy':'R2L','warezclient':'R2L',
    'warezmaster':'R2L','sendmail':'R2L','named':'R2L',
    'snmpgetattack':'R2L','snmpguess':'R2L','xlock':'R2L',
    'xsnoop':'R2L','worm':'R2L',
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R',
    'rootkit':'U2R','httptunnel':'U2R','ps':'U2R',
    'sqlattack':'U2R','xterm':'U2R',
}
df['label'] = df['label'].map(attack_map).fillna('NORMAL')
print(f"  Phan bo nhan:")
for label, count in df['label'].value_counts().items():
    print(f"    {label:<10}: {count:>7}")

# BUOC 3: Ma hoa chu thanh so
print("Dang ma hoa...")
for col in ['protocol_type', 'service', 'flag']:
    df[col] = LabelEncoder().fit_transform(df[col])
le_label    = LabelEncoder()
df['label'] = le_label.fit_transform(df['label'])
print(f"  Nhan: {list(enumerate(le_label.classes_))}")

# BUOC 4: Tao NODE
print("Dang tao node...")
node_keys = df[['protocol_type', 'service']].drop_duplicates().reset_index(drop=True)
node_keys['node_id'] = range(len(node_keys))
df = df.merge(node_keys, on=['protocol_type', 'service'], how='left')
num_nodes = len(node_keys)
print(f"  So node: {num_nodes}")

# BUOC 5: Tao CANH
print("Dang tao canh...")
df['dst_node'] = LabelEncoder().fit_transform(df['flag']) % num_nodes
src_nodes  = df["node_id"].values.astype("int64")
dst_nodes  = df["dst_node"].values.astype("int64")
edge_index = torch.tensor(np.stack([src_nodes, dst_nodes]), dtype=torch.long)
print(f"  So canh: {edge_index.shape[1]}")

# BUOC 6: Tao FEATURE
print("Dang tao features...")
drop_cols    = ['label', 'node_id', 'dst_node']
feature_cols = [c for c in df.columns if c not in drop_cols]
features     = df[feature_cols].values.astype(np.float32)
features     = StandardScaler().fit_transform(features)
edge_attr    = torch.tensor(features, dtype=torch.float)

node_features = np.zeros((num_nodes, features.shape[1]), dtype=np.float32)
node_counts   = np.zeros(num_nodes, dtype=np.float32)
for i, src in enumerate(src_nodes):
    node_features[src] += features[i]
    node_counts[src]   += 1
node_counts   = np.maximum(node_counts, 1)
node_features = node_features / node_counts[:, np.newaxis]
x = torch.tensor(node_features, dtype=torch.float)
print(f"  Node feature shape: {x.shape}")

# BUOC 7: Gan NHAN
print("Dang gan nhan node...")
normal_label = le_label.transform(['NORMAL'])[0]
node_labels  = np.zeros(num_nodes, dtype=np.int64)
for src, label in zip(src_nodes, df["label"].values):
    if label != normal_label:
        node_labels[src] = label
y = torch.tensor(node_labels, dtype=torch.long)
print(f"  Node binh thuong: {(y==0).sum().item()}")
print(f"  Node bi tan cong: {(y!=0).sum().item()}")

# BUOC 8: Chia train/val/test
n   = num_nodes
idx = torch.randperm(n)
train_mask = torch.zeros(n, dtype=torch.bool)
val_mask   = torch.zeros(n, dtype=torch.bool)
test_mask  = torch.zeros(n, dtype=torch.bool)
train_mask[idx[:int(0.6*n)]]            = True
val_mask  [idx[int(0.6*n):int(0.8*n)]] = True
test_mask [idx[int(0.8*n):]]            = True
print(f"  Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

# BUOC 9: Tao PyG Data
data = Data(
    x=x, edge_index=edge_index, edge_attr=edge_attr,
    y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
)
print(f"\nDo thi NSL-KDD tao xong!")
print(f"  So node: {data.num_nodes}")
print(f"  So canh: {data.num_edges}")
print(f"  Feature dim: {data.num_node_features}")

# BUOC 10: Luu
save_dir  = '/content/drive/MyDrive/gnn_models'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'nslkdd_graph.pt')
torch.save({'data': data, 'classes': le_label.classes_}, save_path)
print(f"\nDa luu tai: {save_path}")
