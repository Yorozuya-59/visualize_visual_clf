# プロジェクトの概要
本プロジェクトは，〇〇し，それを〇〇することを目的としています．


# 開発環境に関するヒント
- Ubuntu 22.04 LTS をベースイメージとする Docker コンテナ内
    - 作成されたスクリプトはホストマシン上で git により管理される
    - 複数人で開発することを想定するため，ディレクトリの作成などといった操作はスクリプト内で完結させること
        - 例えば，`os.makedirs(hoge, exist_ok=True)` のような記述をすること
- Python 3.10 を使用
    - `PYTHONPATH=/home/workdir/src` としている
- 開発環境における環境変数は `/home/workdir/.env` に定義されている
    - Python で環境変数をロードするためのライブラリとして `python-dotenv` を使用
    - `figures` / `data` / `logs` / `models` ディレクトリについては，git において管理されないため，原則としてこれらを使用すること
    - また `.env` ファイルに定義されていないディレクトリについては永続化されないため，実験等で必要となる場合は適宜作成しても良いが，その説明と概要は `src/README.md` を作成，もしくは，追記し，記載すること
- 環境に変更があった場合でも汎用的に動作するようにしてください
    - 極力，`os.getenv()` や `glob.glob()` などを使用して，ハードコーディングは避けてください
- 環境の初期化のためのメソッドを `src/mylib/initialize.py` に定義している
    - py スクリプトの冒頭で `from mylib.initialize import init_settings` としてインポートし，`init_settings()` を呼び出すことで環境の初期化を実施すること
    - また，ランダムな挙動を伴う処理では再現性の確保のために `set_seeds` メソッドを利用すること
- 統計計算のライブラリとして Polars を使用
    - Pandas を利用していないため，記法の違いに注意
- データの可視化のためのライブラリとして Matplotlib / Seaborn を使用
    - 原則として Seaborn で解決できることは Seaborn を使用することを推奨


# Python スクリプトのコーディングに関するヒント
- 原則として PEP8 に従うこと
    - 行の長さについては制限を考慮する必要はなく，関数名など，意味が明確になるようにすることを優先すること
    - 変数名についても同様に，意味が明確になるようにすることを優先すること
        - 例えば，for ループについても `i` などの短い変数名を使用するのではなく，意味が明確になるような変数名を使用することを推奨
    - また，ライブラリの推奨記法がある場合にはそれに従うことを推奨
    - ファイルは処理のフローベースで記述し，`if __name__ == '__main__':` や `def main()` といった構造は不要である
        - 上記のような構造は他のスクリプトから呼び出される際の構造であり，このプロジェクトは分析処理ごとにスクリプトを作成するため，外部からの呼び出しは想定しなくて良い
- 関数やクラスについては必ず Docstring を記載すること
    - 体裁は Google スタイルを使用すること
- ライブラリのインポートについては以下のルールで記述すること
    - 大枠は以下の順序で記述すること
        1. 標準ライブラリ
        2. 外部ライブラリ
        3. 自作ライブラリ（`src/mylib` 以下に記述されている処理）
    - インポートするライブラリはブロックごとに記述すること
        - ブロックは機能ごとに区切ること
    - ブロック内でのインポート順序は機能ごととすること
    - `import *` は使用しないこと
    - 条件分岐によるインポートは避けること
    - 冒頭にインポートすること
        - ライブラリのインポートにおいてコメントは不要である
- 適宜，デバッグ時の参考となるようなコメントを `ic` で出力するようにしてください
    - `ic` を用いるのは verbose で容易に切り替えられるためであり，`print` での記述は避けてください
- デバッグ出力の統一
    - デバッグ用の出力には `icecream` の `ic()` を用い、DataFrame の内容出力には `print()` を用いること。
    - 例:  
      ```python
      ic(df_raw.shape)
      ic(df_raw.columns)
      print(df_raw)
      ```
- 何度も呼び出されるような処理のみ関数化するようにしてください
    - 1回，数回しか呼び出されない処理を関数化することは，コードの可読性を下げるためである
- 定数定義の簡素化と一貫性
    - データやファイルパスに関する定数はスクリプト冒頭でまとめて定義すること
    - ディレクトリ名やファイル名は環境変数（`os.getenv()`）や `os.path.join()` を活用し，ハードコーディングを避けること
    - 例:  
      ```python
      LOAD_DATA_DIR = 'raw-data'
      SAVE_DATA_DIR = 'preprocess_raw_data'
      SAVE_FILENAME = 'Log_2025.xlsx'
      OUTPUT_FILENAME = 'processed.parquet'
      load_filepath = os.path.join(os.getenv('DATA_DIR', 'data'), LOAD_DATA_DIR, SAVE_FILENAME)
      save_filepath = os.path.join(os.getenv('DATA_DIR', 'data'), SAVE_DATA_DIR, OUTPUT_FILENAME)
      ```
- 環境変数の利用徹底
    - データディレクトリやファイル名など，環境依存の値は必ず `os.getenv()` で取得し，デフォルト値も明示すること．
    - 例:  
      ```python
      os.getenv('DATA_DIR', 'data')
      ```
- ディレクトリの自動作成
    - ファイル保存前には `os.makedirs(..., exist_ok=True)` で保存先ディレクトリを必ず作成すること


以上の注意点を踏まえたサンプルを以下に示す．

````python
import os 
import glob 
import json 
import yaml

from icecream import ic

import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

from mylib.initialize import init_settings


init_settings(verbose=True)    # ここに load_dotenv() や ic.enable() などの環境の初期化に関する処理を記述している


# Constants
SEED = 42


def example_function(param1: int, param2: str) -> bool:
    '''
    Here is an explanation of the function.

    Args:
        param1 (int): An explanation of the first parameter.
        param2 (str): An explanation of the second parameter.

    Returns:
        bool: An explanation of the return value.

    Raises (Optional):
        ValueError: An explanation of the conditions under which this error is raised.

    Example (Optional):
        Here is an example of how to use this function.

        ```python
        result = example_function(42, 'example')
        ```
    '''
    pass

data_dir = os.getenv('DATA_DIR')
figs_dir = os.getenv('FIGS_DIR')
````



# Polars の利用に関するヒント
- Pandas と記法が異なるため，注意すること
    - また Python の標準記法のように都度変数に渡さずに，メソッドチェーンで記述することを推奨
- DataFrame を格納する変数名については `df_*` とすること
- LazyFrame を格納する変数名についても `df_*` とすること
    - これは LazyFrame と DataFrame が相互に変換可能であるためである
- 開発段階では DataFrame を使用してデバッグを効率的に行い，リファクタリングの段階で LazyFrame に変換することを推奨
- DataFrame を print する際には `ic` ではなく `print` を使用すること
    - DataFrame は icecream の出力では構造が崩れるためである
- Polars のメソッドチェーン活用
    - DataFrame の前処理は，Polars のメソッドチェーンで簡潔に記述すること
    - 例:  
      ```python
      df_raw = (
          pl.read_excel(load_filepath)
          .rename(lambda col_name: col_name.strip().lower())
      )
      ```


# 可視化に関するヒント
- Matplotlib / Seaborn を使用することを推奨
    - 原則として Seaborn で解決できることは Seaborn を使用することを推奨
- 図の保存については `figures` ディレクトリに保存すること
    - また，スクリプト名と同一のディレクトリ以下に保存すること
        - `example_script.py` であれば `figures/example_script/` ディレクトリ以下に保存すること
    - 保存する際のファイル名については，長くなっても図の内容がわかるようなファイル名を使用することを推奨
        - 例えば，`visitor_movement_heatmap.png` のようなファイル名を使用することを推奨


# 分析コードに関するヒント
- 分析コードは原則として，1つの分析について1つのファイルを作成すること
- 分析コードは `src/llm-agents` に作成すること


# PyTorch に関するヒント
- 学習コードについては他のコードと同様に，他から呼び出されることを前提とするようなコーディングは不要である
    - ただし，モデルについては学習時と評価時に呼び出されるため，切り分けて定義するようにするべきである
    - 具体的には `mylib/torch_modules.py` に定義される


# ドキュメント作成に関するヒント
- 基本的に，長文になっても良いので，詳細な説明をするように留意してください
- 構成は以下のものを基本とし，必要に応じて変更してください
    1. 全体概要
    2. ディレクトリ構造
        - 各ディレクトリの簡単な説明
    3. ディレクトリごとの詳説
        - ディレクトリ内の各スクリプトに対して，それぞれがどのような操作を行うものであるかを説明すること
        - ディレクトリ内のスクリプトの依存関係があれば，それを説明すること
        - それぞれのスクリプトから生成される図（`figures/`）やデータ（`data/`）があれば，その説明も記述すること
    4. スクリプトごとの詳説
        - ディレクトリ内の各スクリプトについて詳細に説明すること
        - メソッドやクラスが定義されていれば，それらについて詳細に説明すること
- 可読性の向上のために，必要に応じて Mermaid を利用すること
- ドキュメントの作成時に，用語の定義や意味が曖昧なものはそれらしいことを回答するのではなく，MCP の fetch サーバを用いて検索して，正確な用語の利用に努めること

