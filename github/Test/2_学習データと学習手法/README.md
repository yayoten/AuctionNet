# 2_学習データと学習手法 — 公開データの取得から、学習ベース手法の評価までの検証

学習ベースの手法（BC / IQL / CQL / BCQ / TD3_BC）を、公開データ（約80GB）で学習し直して、シミュレータで評価するまでの道具が、
段階ごとに正しく動くことを pytest で確かめる。前の段階が通ってから、次の段階へ進む。

`github/` そのものの変更（重みの場所・学習ステップ数を引数に出した、など）の回帰テストは、`github/Test/1_初期セットアップ/test_18_model_dir_and_train_args.py` にある。
ここにあるのは、データと、研究側の実行スクリプト（`research/src/`）の検証。

## 手順

```bash
bash github/Test/1_初期セットアップ/setup_env.sh --pip            # .venv（Python 3.9）。済んでいれば不要
.venv/bin/pip install -r research/requirements.txt          # duckdb / pyarrow

bash github/Test/2_学習データと学習手法/download_data.sh           # 1. 公開データを DB/dataset/ に取得・展開（zip 約18GB → 約80GB）
.venv/bin/python research/src/make_train_data.py --workers 2   # 2. 学習データ（rlData）と要約（DB/train_data/）を作る。約20分
.venv/bin/python research/src/train_model.py <spec.json>       # 3. 学習。重みは DB/models/<model_id>/ に入る
.venv/bin/python research/src/run_experiment.py <spec.json>    # 4. 評価。player.kwargs.model_dir に DB/models/<model_id> を書く

cd github/Test/2_学習データと学習手法
../../../.venv/bin/python -m pytest                    # 全部
../../../.venv/bin/python -m pytest -m "not needs_data"   # 80GB の原本が無い端末で走るものだけ
```

## ファイル

| ファイル | 段階 | 検証内容 |
|---|---|---|
| download_data.sh | 1 | 取得と展開（再実行で続きから。macOS の `._*.csv` は展開しない） |
| test_01_raw_dataset | 1 | 21 日分がそろい、列が仕様の 18 列。欠損なし、48 社 × 約 50 万 PV、3 枠が毎回落札されている。広告主の予算・CPA 制約・カテゴリがシミュレータの 48 社と同じ |
| test_02_train_data | 2 | rlData が 21 × 48 × 48 行、状態 16 次元が有限、時間・予算・PV 数の特徴量の整合、done と next_state の対応、記録した SHA-1 との一致 |
| test_03_train_driver | 3 | `train_model.py`：5 手法が学習でき、条件・重みの SHA-1・損失が記録される。既定のステップ数は本家の値。同じ条件なら同じ重みになる（乱数の固定）。失敗は記録して止めない。spec の `train_kwargs`（本家の学習関数に渡す追加の引数。例：BCQ の `max_action`）が渡り、meta に残り、model_id に入る（書かないときの ID は従来どおり） |
| test_04_eval_driver | 4 | `run_experiment.py`：同梱の重みと自前の重みの両方を、縮小設定で評価できる。どの重みを使ったかが run に残る。ルールベースもこれまでどおり動く |
| test_05_trained_models | 5 | `DB/models/` の本番の重み：記録どおりのファイル、コミット済みのコードで CPU 学習、損失が有限、戦略として読めて入札が有限。既定設定の重みは、学習し直すと同じ重み（テンソルのハッシュが一致）・同じ損失になる（model_id の一致は、`github/` の版が学習時と同じときだけ比べる。`github/` を変えると ID は変わるが、重みは変わらないことを確かめる） |
| test_99_untouched | — | テストが `github/` と `DB/` を汚していない |

`needs_data` の印が付いたテストは、`DB/dataset/traffic/` が無い端末ではスキップする。原本の中身（行数、欠損、広告主の並び）は、
学習データを作るときに書き出した要約（`DB/train_data/period-N.json`、git 管理）で確かめるので、原本が無くても走る。

## 置き場所

| もの | 場所 | git |
|---|---|---|
| 原本（period-7〜27.csv）と zip | `DB/dataset/traffic/`、`DB/dataset/zip/` | 管理外（約80GB＋18GB） |
| 学習データ（rlData） | `DB/dataset/traffic/training_data_rlData_folder/` | 管理外（原本から作り直せる） |
| 原本の要約、学習データの SHA-1 | `DB/train_data/` | 管理 |
| 学習した重み・条件・損失 | `DB/models/<model_id>/` | 管理（1 つ数百KB） |

原本を `data/` ではなく `DB/dataset/` に置くのは、`github/Test/1` の `test_16` が「リポジトリ直下に `data/traffic` が無い」ことを前提にしているため。

## 分かったこと（2026-10-08、Linux x86_64 / Python 3.9.16 / torch 1.12.0）

- **GPU は使っていない。** ロックの `torch==1.12.0`（Linux の PyPI 版は CUDA 10.2）は RTX 3090 に対応せず、CUDA の計算が
  `no kernel image is available` で落ちる。`torch.cuda.is_available()` は True を返すので、学習コードは GPU を選んでしまう。
  テストと実行スクリプトは、`CUDA_VISIBLE_DEVICES=""` で GPU を隠して CPU で動かす（REP001 の Windows も CPU）。
- **uv の新しい版では `setup_env.sh` の uv 経路が通らない**（`gin==0.1.6` の配布物のファイル名を拒否する）。`--pip` を付ける。
- **同梱の BCQ の重みと、リポジトリの学習コードが保存する BCQ は、呼び出しの形が違う**（同梱は `forward(states, eval_flag)`、
  学習コードは `forward(states)`）。シミュレータ側の BCQ 戦略は `eval_flag=True` を渡すので、自前の重みを呼べなかった。
  保存された形を見て呼び分けるように直した（`github/simul_bidding_env/strategy/bcq_bidding_strategy.py`）。
- **本家の `main_*.py` は `random` の乱数を固定していない**（`ReplayBuffer.sample` は `random.sample`）。そのまま実行すると、
  学習のたびに重みが変わる。`train_model.py` は `random` / numpy / torch の 3 つを固定し、同じ条件で同じ重みになることをテストで確かめている。
- **重みが同じでも、保存ファイル（torch.jit）の SHA-1 は一致しないことがある。** ファイルには、同じプロセスでそれまでに保存した
  モデルの数で変わる連番（`___torch_mangle_N`）が入る。コマンドから 1 つずつ学習し直せばファイルの SHA-1 も一致するが、
  pytest の中（先に別のモデルを保存している）では一致しない。重みが同じかは、テンソルそのもののハッシュ（`train_model.weights_sha1`）で比べる。
- **学習時と推論時で、状態の正規化の式がわずかに違う**（学習は `(x−min)/(max−min+0.01)`、推論は `(x−min)/(max−min)`）。
  対象は PV 数の 3 つの特徴量で、範囲が数万あるので差は小さい。直していない。
