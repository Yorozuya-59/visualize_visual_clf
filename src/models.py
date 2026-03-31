import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 以前のモデル (そのまま残します) ---
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_activations(self, x):
        act1 = F.relu(self.conv1(x))
        act2 = F.relu(self.conv2(F.max_pool2d(act1, 2)))
        return act1, act2

# --- 新しい高精度化モデル (チャンネル増、BatchNorm、Dropout追加) ---
class AdvancedCNNNet(nn.Module):
    def __init__(self):
        super(AdvancedCNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = F.relu(self.bn3(x))
        x = self.dropout2(x)
        
        x = self.fc2(x)
        return x

    def get_activations(self, x):
        act1 = F.relu(self.bn1(self.conv1(x)))
        act2 = F.relu(self.bn2(self.conv2(self.pool(act1))))
        return act1, act2
