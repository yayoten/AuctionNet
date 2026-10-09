# 1_初期セットアップ — AuctionNet（`github/`）が動くことの検証

クローン直後の `github/` は README の手順どおりには動かなかった。不具合を `github/` 側で修正し、その修正が効いていること・全体が正しく動くことを pytest で検証する。テストの生成物はすべて tmp に出す（`test_99` が、テスト実行が `github/` を汚していないことを保証）。

## 実行

```bash
bash github/Test/1_初期セットアップ/setup_env.sh        # リポジトリ直下に .venv（Python 3.9）を作る
cd github/Test/1_初期セットアップ
../../../.venv/bin/python -m pytest                  # 全部（約8分）
../../../.venv/bin/python -m pytest -m "not slow"    # 速いものだけ（約3分）
../../../.venv/bin/python -m pytest -m "not full_scale"
```

本体の実行（README の手順のまま。`PYTHONPATH` は不要）:

```bash
cd github && ../.venv/bin/python main_test.py     # 約100秒
```

最終結果（2026-10-07、macOS arm64 / Python 3.9.6 / torch 1.12.0 / numpy 1.24.2）: **1049 passed, 1 skipped, xfail 0**。
2026-10-08、Linux x86_64（RTX 3090 搭載）/ Python 3.9.16 / torch 1.12.0+cu102、`setup_env.sh --pip`: **1079 passed, 9 skipped**（test_18 を追加。スキップは、macOS で取った基準値との比較 8 件と、BCQ の飽和 1 件）。
2026-10-09、同じ端末、test_19 を追加：**1083 passed, 9 skipped**（全件を回した回は、別の作業が実行中に `github/Test/README.md` を編集したため、`github/` の無変更を見る 2 件だけ落ちた。その 2 件は、回し直して通った）。

## 別の端末で動かす

1. リポジトリを取得する（`.venv/` は git 管理外なので、端末ごとに作る）。
2. `bash github/Test/1_初期セットアップ/setup_env.sh` を実行する。
   - `uv` があれば uv を使う（Python 3.9 が入っていなくても uv が取得する）。
   - `uv` が無ければ、手元の Python 3.9 + pip で作る。どちらも無いときは、用意する方法を表示して止まる。
   - 既定で `requirements.lock.txt`（全 31 パッケージのバージョン固定）を入れるので、どの端末でも同じ版になる。
3. `cd github/Test/1_初期セットアップ && ../../../.venv/bin/python -m pytest` で確認する。

| オプション | 用途 |
|---|---|
| `--pip` | uv があっても使わず、Python 3.9 + pip で作る |
| `--no-lock` | ロックを使わず、`github/requirements.txt` から解決する |
| `作成先DIR` | `.venv` 以外の場所に作る |

**端末が違うときの注意**

- **Python は 3.9 が必須。** `torch==1.12.0` は 3.10 以降に配布が無い。
- **基準値（`golden_full_scale.json`）は macOS arm64 で取得した値。** OS や CPU が違うと乱数・浮動小数の実装差でずれうるので、違う端末では基準値との比較だけ自動でスキップする（理由を表示）。動作の検証と「同じ端末で 2 回回して同じ結果になる」ことの確認は、どの端末でも走る。その端末の基準値を作るには `UPDATE_GOLDEN=1 ../../../.venv/bin/python -m pytest test_14_full_scale.py`。
- **依存を変えたらロックを作り直す。** `uv pip freeze --python .venv/bin/python > github/Test/1_初期セットアップ/requirements.lock.txt`（`test_01` が、requirements・ロック・実環境の 3 つの一致を確認する）。
- **確認済みなのは macOS arm64 だけ。** uv 経路・pip 経路・`--no-lock` の 3 通りで環境を作り、テストが通ることを確認した。Linux / Windows / GPU 搭載機では未確認（下記）。
  - Windows は Git Bash か WSL で実行する想定（PowerShell 用のスクリプトは無い）。
  - GPU 搭載機では、学習側（BC・IQL・CQL・TD3_BC・BCQ）が自動で CUDA を使う。テストは CPU 前提なので、`conftest.py` が `CUDA_VISIBLE_DEVICES=""` で GPU を隠す（Linux + RTX 3090 で確認）。
  - Linux では PyPI の `torch==1.12.0` が CUDA 10.2 同梱版になる。RTX 3090 のような新しい GPU には対応しておらず、GPU で計算すると `no kernel image is available` で落ちる（`torch.cuda.is_available()` は True を返す）。
  - uv 0.12 では、`gin==0.1.6` の配布物のファイル名が拒否されて uv 経路が通らない。`setup_env.sh --pip` を使う（Linux で確認）。

## ファイル

| ファイル | 検証内容 |
|---|---|
| test_01_environment | Python / 依存バージョン、gin 同居、torch.jit、requirements の網羅性 |
| test_02_layout | ファイル構成、同梱モデル（7戦略）・normalize_dict・OnlineLP CSV・PV モデルの実在と読込 |
| test_03_imports | 全 .py（約60）の import、副作用なし、クリーンなインタプリタでの import |
| test_04_gin_config | test.gin の値、読み込めること、値がコードに届くこと |
| test_05 / 05b | PV 生成（NeurIPSPvGen / ModelPvGenerator）の形・範囲・再現性 |
| test_06_bidding_env | オークション（第2価格・slot・露出・CV・reserve）を手計算＋不変条件で |
| test_07 / 08 | ルールベース戦略（PID/ABid/OnlineLP）、学習済み戦略7種 |
| test_09 | PlayerAgentWrapper（転送・2秒タイムアウト） |
| test_10 | BiddingTracker（ログCSV）、PlayerAnalysis（スコア式） |
| test_11 / 12 | Controller（48体の構成・予算/CPA）、run_test の補助関数 |
| test_13 | 縮小設定（2000PV）の通し実行：予算制約・ログ整合・再現性 |
| test_14 | **README の手順そのまま** `cd github && python main_test.py`（500000PV・2エピソード・2プレイヤー、約100秒、基準値 `golden_full_scale.json`） |
| test_15 | 学習側：BC/IQL/CQL/BCQ/TD3_BC/DT、共通utils、オフライン環境 |
| test_16 | シミュレータ → 学習データ → 学習 → 保存 → 戦略として読込 → オフライン評価 の通し |
| test_17 | 直書きだった値を引数に出した変更（PID / ABid / OnlineLP / 環境 / スコア式）の回帰 |
| test_18 | 学習ベース 5 戦略の `model_dir`（既定は同梱の重み）、学習スクリプトの `train_data_path` / `save_path` / `step_num`（既定は従来の値）、BCQ の 2 つの保存形式 |
| test_19 | BCQ の学習の入口（`run_bcq.train_bcq_model`）の `max_action`（既定は従来の 100。渡さないときと同じ重みになる。渡すと、生成モデルと方策の上限がその値になる）。W001/REP003 のループ 3 で追加 |
| test_99 | `github/` が無変更であること |

## 結論

修正後は、README の手順のまま動く。`cd github && python main_test.py` は終了コード 0 で完走し、結果は再現する（基準値: `rank_score = 0.001895`、player 0/1 の reward = 46 / 50）。学習済みモデルが無い間は、プレイヤーは PID にフォールバックする。

## `github/` に入れた修正（19 ファイル）

| # | クローン直後の不具合 | 修正 | 検証 |
|---|---|---|---|
| 1 | `config/test.gin` が `Ambiguous selector 'run_test'` で読めず、`main_test.py` が起動しない | gin の import を `github.` 付きに。`main_test.py` は自分でリポジトリ直下を sys.path に入れ、gin ファイルもスクリプト位置から解決 | test_04, test_14 |
| 2 | 学習済みモデルが無いと `sys.exit(1)` | `initialize_player_agent` は読み込めなければ理由をログに出して PID にフォールバック | test_12, test_14 |
| 3 | `requirements.txt` に `einops` が無い／`matplotlib==3.3.4` が arm64 でビルド不可 | `einops==0.8.2` を追加、`matplotlib==3.9.4` に | test_01 |
| 4 | `NeurIPSPvGen.reset()` が `pv_num` 等を既定値に戻す。gin の `PVNUM` は未束縛 | reset で設定を保持。`Controller.pv_num = %PVNUM` を束縛 | test_05, test_13 |
| 5 | CV の乱数が pValue ノイズと同じ固定シードを共有（pValue≈0.5 で CV率0.17） | CV 用に別シード（`CONVERSION_SEED`） | test_06 |
| 6 | PID の `reset()` が `last_remaining_budget` を戻さない。プレイヤー戦略は構築時に reset されない | PID の reset を修正。Controller がプレイヤーも reset | test_07, test_11 |
| 7 | 生成ログの `pvIndex` が tick ごとに 0 から | エピソード内の通し番号に | test_13 |
| 8 | `from collections import Iterable`（Python 3.10 以降で不可） | `collections.abc` に | test_12 |
| 9 | Controller の既定値が 30体/24tick（環境は48固定）。`modelPvGen` に `pv_num` が渡らない | 既定値を 48tick・8×6 に。`pv_num` を渡す | test_04, test_11, test_05b |
| 10 | DT の `load_net` がディレクトリ不可、`save_jit` が分かりにくい RuntimeError。`offline_env.test()`・`NeurIPSPvGen.test()`・`utils` のデモが壊れている | `load_net` はディレクトリ可、`save_jit` は `NotImplementedError`。デモを修正 | test_15, test_05 |
| 11 | `python main/main_bc.py` 等が `No module named 'github'` | `main_*.py` がリポジトリ直下を sys.path に入れる | test_03 |

**数値が変わる修正に注意**: #5（CV の乱数）と #6（PID）は、シミュレーション結果そのものを変える。クローン直後の挙動（gin だけ迂回）では `rank_score = 0.000863`、reward = 31 / 50 だった。本家の公開値や論文の数値と比べるときは、この差を踏まえること。

## 仕様として固定している挙動（修正していない）
- 入札者が1人だけだと、市場価格が reserve と等しくなり、落札扱いにならない。
- 露出の乱数は固定シードで、同じ形の入力なら tick が違っても同じ並びになる。
- PlayerAnalysis はデータが空でも例外にならず NaN を返す。
- BCQ の同梱モデルは、テストの入力範囲では出力が 95 で一定（状態に反応していないように見える。未調査）。
- `main_*.py` を import すると sys.path に `strategy_train_env` が足される（conftest で毎テスト取り除いている）。
- `main_onlineLp.py` はデータが無いと何もせず正常終了する。

## Windows（2026-10-07、Windows 11 AMD64 / Python 3.9.25 / CPU 版 torch 1.12.0）

`setup_env.sh`（uv 経路）で環境を作り、全テストが通ることを確認した（1040 passed, 9 skipped。skip は macOS 基準値との比較など）。直した点：

- `github/config/test.gin` 27行目のコメントの全角括弧を ASCII に。Windows の既定 cp932 で gin が読めず `main_test.py` が起動しなかった。
- テスト側：`read_text()` に `encoding="utf-8"` を指定。`saved_model/IQLtest/...` のパス区切りに依存した比較をやめた。`github/results/`（git に載らない空ディレクトリ）が無くても通るようにした。
