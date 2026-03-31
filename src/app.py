import gradio as gr
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import os
from models import *

# 1. モデルのセットアップ
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AdvancedCNNNet().to(device)

model_path = os.getenv('MODELS_DIR', '/home/workdir/models') + '/mnist_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # 推論モード

# 2. 画像前処理の定義
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 3. 推論およびヒートマップ生成関数
def predict_digit(sketch):
    if sketch is None:
        return None, None
    
    # Gradio の Sketchpad は dict を返す場合があるため対応
    if isinstance(sketch, dict):
        img_arr = sketch.get('composite', sketch.get('background'))
    else:
        img_arr = sketch

    # PIL画像に変換してグレースケール化
    image = Image.fromarray(img_arr).convert('L')
    # モデルの入力サイズに合わせて28x28にリサイズ
    image = image.resize((28, 28))
    img_np = np.array(image)
    
    # Gradioのキャンバスは白背景に黒ペンであることが多いため、
    # MNIST（黒背景に白文字）に合わせて色を反転させる
    if img_np.mean() > 127:
        img_np = 255 - img_np
        
    # PyTorchテンソルに変換: [H, W] -> [1, 1, 28, 28]
    image_tensor = transform(img_np).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # クラス分類の推論
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        # GradioのLabelコンポーネント用に辞書を作成
        prob_dict = {str(i): probs[i].item() for i in range(10)}
        
        # 中間層の特徴マップを取得
        act1, act2 = model.get_activations(image_tensor)
        
    # 特徴マップの可視化処理 (チャンネルごとの出力を平均して2次元のヒートマップ化)
    map1 = act1[0].mean(dim=0).cpu().numpy()
    map2 = act2[0].mean(dim=0).cpu().numpy()
    
    # Matplotlibで図表を作成
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(map1, cmap='viridis')
    axes[0].set_title("Conv Layer 1 (Low-level features)")
    axes[0].axis('off')
    
    axes[1].imshow(map2, cmap='viridis')
    axes[1].set_title("Conv Layer 2 (High-level features)")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # 確率の辞書と、生成した図表(fig)を返す
    return prob_dict, fig

# 4. Gradio UIの構築
# 【修正点1】theme 引数を削除
with gr.Blocks() as demo:
    gr.Markdown("# ✍️ MNIST CNN リアルタイム推論 & 特徴マップ可視化")
    gr.Markdown("左のキャンバスに数字（0〜9）を描くと、モデルの予測確率と内部（畳み込み層）の反応領域が右側に表示されます。")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 【修正点2】crop_size 引数を削除
            inp = gr.Sketchpad(label="キャンバス", type="numpy")
            btn = gr.Button("推論する", variant="primary")
            
        with gr.Column(scale=1):
            # 出力コンポーネント
            out_label = gr.Label(num_top_classes=3, label="AIの予測 (Top 3)")
            out_img = gr.Plot(label="中間層の反応ヒートマップ")
            
    # ボタンクリック時のイベント紐付け
    btn.click(fn=predict_digit, inputs=inp, outputs=[out_label, out_img])

if __name__ == "__main__":
    # 【修正点3】theme 引数をこちらに移動
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
