import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os

# 1. デバイスの設定
# docker-compose.yml の 'capabilities: [gpu]' 設定により、自動的にCUDA(GPU)が選択されます
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. データセットの準備と前処理
# ホスト側の ./data と同期しているマウント先ディレクトリを指定
data_dir = '/home/workdir/data'

# 画像をテンソルに変換し、[-1, 1]の範囲に正規化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNISTデータセットのダウンロードと読み込み
print("Loading dataset...")
train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. モデルの定義 (シンプルな多層パーセプトロン)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet().to(device)

# 4. 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 学習ループ
epochs = 5
print("Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    # Dockerfileでインストール済みの tqdm を活用して進捗を可視化
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for images, labels in progress_bar:
        # データをGPU（またはCPU）へ転送
        images, labels = images.to(device), labels.to(device)

        # 勾配の初期化
        optimizer.zero_grad()

        # 順伝播 -> 損失計算 -> 逆伝播 -> パラメータ更新
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {running_loss/len(train_loader):.4f}")

# 6. 学習済みモデルの保存
# ホスト側の ./models と同期しているマウント先ディレクトリを指定
model_dir = '/home/workdir/models'
os.makedirs(model_dir, exist_ok=True)

save_path = os.path.join(model_dir, 'mnist_model.pth')
torch.save(model.state_dict(), save_path)
print(f"Training complete. Model saved to {save_path}")
