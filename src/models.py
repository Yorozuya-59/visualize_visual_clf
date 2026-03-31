import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        # 1入力チャンネル(グレースケール), 16出力チャンネル, 3x3カーネル
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # 16入力チャンネル, 32出力チャンネル, 3x3カーネル
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # 28x28の画像は、2回のプーリング(2x2)を経て 7x7 に縮小されます
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # 全結合層へ渡すために1次元に平坦化
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_activations(self, x):
        """
        推論時に、各層が画像の「どこ」に反応したか（特徴マップ）を取得するための独自メソッド
        """
        # 1層目の特徴マップ
        act1 = F.relu(self.conv1(x))
        # 2層目の特徴マップ (1層目の出力をプーリングしてから2層目へ)
        act2 = F.relu(self.conv2(F.max_pool2d(act1, 2)))
        return act1, act2
