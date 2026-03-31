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
    # 一時変数には `_` を付与してスコープを限定
    _data_dir = os.getenv('DATA_DIR')

    if not _data_dir:
        mo.stop(True, "環境変数 'DATA_DIR' が設定されていません。./environment/.env ファイルを確認してください。")

    print(f"Data directory: {_data_dir}")

    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    _train_dataset = torchvision.datasets.MNIST(root=_data_dir, train=True, download=True, transform=_transform)
    _test_dataset = torchvision.datasets.MNIST(root=_data_dir, train=False, download=True, transform=_transform)

    # DataLoaderは他セルで使うため public のまま
    train_loader = torch.utils.data.DataLoader(_train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(_test_dataset, batch_size=64, shuffle=False)
    return test_loader, train_loader


@app.cell
def _(train_loader):
    import matplotlib.pyplot as plt
    import numpy as np

    # 他セル(推論確認)でも使うため関数として public に定義
    def imshow(img):
        img = img / 2 + 0.5  
        npimg = img.numpy()
        return np.transpose(npimg, (1, 2, 0))

    # 描画用の一時変数はすべてプライベート化
    _dataiter = iter(train_loader)
    _image_batch, _label_batch = next(_dataiter)

    _num_images = 40
    _rows, _cols = 4, 10
    # 10列表示に合わせて横のサイズを大きく設定
    _fig_data, _axes_data = plt.subplots(_rows, _cols, figsize=(20, 8))

    # 2次元配列のaxesを1次元に平坦化してループ処理
    _axes_flat = _axes_data.flatten()

    for _idx in range(_num_images):
        _axes_flat[_idx].imshow(imshow(_image_batch[_idx]), cmap='gray')
        _axes_flat[_idx].set_title(f"Label: {_label_batch[_idx].item()}")
        _axes_flat[_idx].axis('off')

    _fig_data.tight_layout()

    # Marimo描画用
    _fig_data
    return imshow, plt


@app.cell
def _(device, summary):
    from models import CNNNet

    # SimpleNet から CNNNet へ変更
    model = CNNNet().to(device)

    _model_summary = summary(model, input_size=(64, 1, 28, 28))
    _model_summary
    return (model,)


@app.cell
def _(device, model, nn, optim, tqdm, train_loader):
    # 学習ループ内の変数は他セルから隠蔽する
    _criterion = nn.CrossEntropyLoss()
    _optimizer = optim.Adam(model.parameters(), lr=0.001)

    _epochs = 5
    print("Starting training...")

    for _epoch in range(_epochs):
        model.train()
        _running_loss = 0.0

        _progress_bar = tqdm(train_loader, desc=f"Epoch {_epoch+1}/{_epochs}")
        for _images, _labels in _progress_bar:
            _images, _labels = _images.to(device), _labels.to(device)

            _optimizer.zero_grad()

            _outputs = model(_images)
            _loss = _criterion(_outputs, _labels)
            _loss.backward()
            _optimizer.step()

            _running_loss += _loss.item()
            _progress_bar.set_postfix(loss=_loss.item())

        print(f"Epoch [{_epoch+1}/{_epochs}] Average Loss: {_running_loss/len(train_loader):.4f}")
    return


@app.cell
def _(model, os, torch):
    _model_dir = os.getenv('MODELS_DIR')

    # Noneだった場合の安全策を追加
    if not _model_dir:
        _model_dir = '/home/workdir/models'

    os.makedirs(_model_dir, exist_ok=True)
    _save_path = os.path.join(_model_dir, 'mnist_model.pth')

    torch.save(model.state_dict(), _save_path)
    print(f"Training complete. Model saved to {_save_path}")
    return


@app.cell
def _(device, imshow, model, plt, test_loader, torch):
    model.eval()

    _dataiter_test = iter(test_loader)
    _images_test, _labels_test = next(_dataiter_test)
    _images_test_device = _images_test.to(device)

    with torch.no_grad():
        _preds = model(_images_test_device)
        _, _predicted = torch.max(_preds, 1)

    _num_display = 40
    _rows_pred, _cols_pred = 4, 10
    # 10列表示に合わせて横のサイズを大きく、タイトル2行分のため縦も調整
    _fig_pred, _axes_pred = plt.subplots(_rows_pred, _cols_pred, figsize=(20, 10))

    # 2次元配列のaxesを1次元に平坦化してループ処理
    _axes_pred_flat = _axes_pred.flatten()

    for _idx in range(_num_display):
        _img = imshow(_images_test[_idx])
        _axes_pred_flat[_idx].imshow(_img, cmap='gray')

        _true_label = _labels_test[_idx].item()
        _pred_label = _predicted[_idx].item()

        _text_color = "black" if _true_label == _pred_label else "red"
        _axes_pred_flat[_idx].set_title(f"Pred: {_pred_label}\nTrue: {_true_label}", color=_text_color)
        _axes_pred_flat[_idx].axis('off')

    _fig_pred.tight_layout()

    _fig_pred
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
