import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential as GNNSequential
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

print(f"資料集: {dataset.name}")
print("===================================")
print(f"圖的數量: {len(dataset)}")
print(f"節點特徵維度: {dataset.num_node_features}")
print(f"分類類別數: {dataset.num_classes}")

# --- 切分資料集 ---
torch.manual_seed(42) 
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f"\n訓練集大小: {len(train_dataset)}")
print(f"測試集大小: {len(test_dataset)}")

# --- 建立 DataLoader ---
# DataLoader 會自動處理批次化，將多張小圖合併成一張包含許多獨立 компоненты 的大圖
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- 模型定義 (與之前相同，無需修改) ---
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels = None):
        super().__init__()
        if not hidden_channels:
            hidden_channels = out_channels
        self.net = nn.Sequential(
            nn.Linear(in_channels,hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)

class GIN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.nn1 = MLP(in_channels, hidden_channels)
        self.nn2 = MLP(hidden_channels, out_channels, hidden_channels)

        self.conv1 = GINConv(self.nn1)
        self.conv2 = GINConv(self.nn2)

        self.norm1 = nn.BatchNorm1d(hidden_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.drop = nn.Dropout(0.5)
        # self.pool = nn.global_add_pool 

        self.res_proj1 = nn.Identity() if in_channels == hidden_channels else nn.Linear(in_channels, hidden_channels)
        self.res_proj2 = nn.Identity() if hidden_channels == out_channels else nn.Linear(hidden_channels, out_channels)


    def forward(self, x, edge_index, batch_index):
        res = self.res_proj1(x)

        x = self.conv1(x,edge_index)
        x = self.norm1(x)
        x = F.relu(res + x)
        x = self.drop(x)

        res = self.res_proj2(x)

        x = self.conv2(x,edge_index)
        x = self.norm2(x)
        x = F.relu(res + x)
        x = self.drop(x)

        x = global_add_pool(x, batch_index)
        out = F.log_softmax(x, dim=1)

        return out


# --- 初始化模型、優化器和損失函數 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(dataset.num_node_features, dataset.num_classes, 64).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01,weight_decay = 5e-2)
criterion = nn.NLLLoss()


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output, data.y) 
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            pred = output.argmax(dim=1)
            correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

# --- 7. 執行訓練循環 ---
print("\n--- 開始訓練 ---")
for epoch in range(1, 101):
    train_loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if epoch % 10 == 0:
      print(f'Epoch: {epoch:03d}, train_loss: {train_loss:.4f}, '
            f'train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}')

print("--- 訓練完成 ---")