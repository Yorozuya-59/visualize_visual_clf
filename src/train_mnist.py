import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torchinfo import summary
    from tqdm import tqdm
    from dotenv import load_dotenv

    from mylib.initialize import init_settings, set_seeds


    init_settings(verbose=True)


    # マウントされている .env ファイルから環境変数を読み込む
    load_dotenv()

    # GPUが利用可能か自動判定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device, nn, optim, os, summary, torch, torchvision, tqdm, transforms


@app.cell
def _(mo, os, torch, torchvision, transforms):
    # 環境変数からデータ保存先のパスを取得
    data_dir = os.getenv('DATA_DIR')

    if not data_dir:
        mo.stop(True, "環境変数 'DATA_DIR' が設定されていません。./environment/.env ファイルを確認してください。")

    print(f"Data directory: {data_dir}")

    # 画像のテンソル変換と正規化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader, train_loader


@app.cell
def _(train_loader):
    import matplotlib.pyplot as plt
    import numpy as np
    # import torchvision

    # 正規化されたテンソルを元の画像データ（[0, 1]の範囲）に戻す関数
    def imshow(img):
        img = img / 2 + 0.5  # [-1, 1] を [0, 1] に戻す
        npimg = img.numpy()
        return np.transpose(npimg, (1, 2, 0))

    # Dataloaderから1バッチ分のデータを取得
    dataiter = iter(train_loader)
    image_batch, label_batch = next(dataiter)

    # 先頭の8枚を抽出して描画
    num_images = 8
    fig_data, axes_data = plt.subplots(1, num_images, figsize=(12, 3))

    for idx_i in range(num_images):
        # 画像はモノクロ（グレースケール）で表示
        axes_data[idx_i].imshow(imshow(image_batch[idx_i]), cmap='gray')
        axes_data[idx_i].set_title(f"Label: {label_batch[idx_i].item()}")
        axes_data[idx_i].axis('off')

    fig_data.tight_layout()

    # セルの最後にFigureオブジェクトを置くことでMarimo上に表示させます
    fig_data
    return imshow, plt


@app.cell
def _(device, nn, summary):
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

    # torchinfoを利用したモデルアーキテクチャの出力
    # (バッチサイズ64, チャンネル数1, 画像サイズ28x28)
    model_summary = summary(model, input_size=(64, 1, 28, 28))
    model_summary
    return (model,)


@app.cell
def _(device, model, nn, optim, tqdm, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    print("Starting training...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {running_loss/len(train_loader):.4f}")
    return


@app.cell
def _(model, os, torch):
    model_dir = os.getenv('MODELS_DIR')
    save_path = os.path.join(model_dir, 'mnist_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")
    return


@app.cell
def _(device, imshow, model, plt, test_loader, torch):
    # モデルを推論モードに切り替え
    model.eval()

    # テストデータから1バッチ取得
    dataiter_test = iter(test_loader)
    images_test, labels_test = next(dataiter_test)

    # 推論を行うため、画像をデバイス（GPU/CPU）へ転送
    images_test_device = images_test.to(device)

    # 勾配計算を無効化して推論を実行
    with torch.no_grad():
        preds = model(images_test_device)
        # クラスごとの確率（ロジット）が最も高いインデックスを予測値として取得
        _, predicted = torch.max(preds, 1)

    # 結果の可視化（先頭の8枚）
    num_display = 8
    fig_pred, axes_pred = plt.subplots(1, num_display, figsize=(14, 3))

    for idx_j in range(num_display):
        # 画像の描画 (描画用にCPU上のデータを使用)
        img = imshow(images_test[idx_j])
        axes_pred[idx_j].imshow(img, cmap='gray')

        true_label = labels_test[idx_j].item()
        pred_label = predicted[idx_j].item()

        # 予測が正解なら黒文字、不正解なら赤文字でタイトルを表示
        text_color = "black" if true_label == pred_label else "red"
        axes_pred[idx_j].set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=text_color)
        axes_pred[idx_j].axis('off')

    fig_pred.tight_layout()

    # Marimo上に表示
    fig_pred
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
