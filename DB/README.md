# DB/ — 実験結果のデータベース

すべての実験結果を、ここに溜める。REP ごとに別々に持たない。
後の検証が、過去の結果（どのパラメータで、どんな結果だったか）を、REP を開かずに SQL で引けるようにするため。
人が直接読むことは想定しない。読みたいときは Claude に頼み、必要な部分を抜き出して見せてもらう。

## 構造

```
DB/
├── README.md              # このファイル
├── 記録の一覧.md           # 何を、なぜ、どの段で取るか。段ごとの容量と時間の実測。取らない量とその理由
├── make_columns.py        # 列の辞書（意味・単位・なぜ取るか）を書く。ここが正
├── columns.json           # 上の生成物
├── build_db.py            # runs/ から DuckDB を作り直す
├── query.py               # SQL を投げて表示する（Claude 用）
├── auctionnet.duckdb      # 生成物（git 管理外）。いつ消して作り直してもよい
├── runs/                  # 原本（追記のみ。手で編集しない）
│   └── R{ハッシュ}-p{位置}/        # 1 実行 = 1 つの設定 × プレイヤー位置 1 つ
│       ├── params.json     # spec（人が書いた値）と effective（既定値を補った、実際に使われた値）
│       ├── meta.json       # コミット・端末・版・所要時間・status
│       ├── episodes.parquet / ticks.parquet / agents.parquet
│       ├── sim_iters.parquet            # 記録の形式の版 2 から。予算超過のやり直しの各回
│       └── raw/                         # 記録の形式の版 2 から。大きい記録（git 管理外）
│           ├── agent_ticks.parquet      # ティック × 全 48 社の集計と、手法の内部状態（段 standard 以上）
│           ├── pv_won.parquet           # プレイヤーが落札した機会ごと（段 standard 以上）
│           ├── pv_all.parquet           # プレイヤーの全機会（段 detail 以上）
│           └── bids_all_ep{N}.parquet   # 全 48 社の、機会ごとの入札額（段 full）
├── models/                # 学習した重みの原本（追記のみ）
│   └── M{ハッシュ}/               # 1 つの学習 = 手法 × ステップ数 × 乱数シード × 学習データ × コードの版
│       ├── *_model.pth / normalize_dict.pkl   # 重み（torch.jit）と、状態の正規化の値
│       ├── meta.json       # 条件・コミット・端末・所要時間・重みの SHA-1・status
│       └── loss.csv        # 1 ステップごとの損失
└── train_data/            # 公開データ（DB/dataset/、git 管理外）の要約と、学習データの SHA-1
    ├── period-N.json       # 1 日分の行数・欠損・広告主ごとの予算／CPA 制約／平均の α など
    └── training_data_all.json
```

- **原本は `runs/`、`.duckdb` は索引。** 並列実行しても衝突しないよう、実行ごとに別フォルダへ書く。
- **`params.json` と結果は同じフォルダにある。** どの設定でどの結果が出たかが、常に対応している。
- **run_id は「設定＋コードの版」のハッシュ。** 同じ設定を同じコードで流すと同じ run_id になり、既にあればスキップする（`--force` で上書き）。コード（`github/`）が変わると run_id が変わるので、古いコードの結果と取り違えない。
- 未コミットの変更がある状態で流すと、`runs.github_dirty = true` になる。その結果はコミットから再現できないので、論文・報告に使う前に、コミットしてから流し直す。

## 記録の形式の版と、記録の段（2026-10-10、T006）

実験を流し直さずに済むよう、実験のたびに残す量を増やした。**何を、なぜ取るか、段ごとの容量と時間は `記録の一覧.md`。**

| 版 | いつ | 中身 |
|---|---|---|
| 1 | 2026-10-09 まで（1,525 run） | `episodes`・`ticks`・`agents` だけ。**書き換えない。読むだけ** |
| 2 | 2026-10-10 から | 上の 3 つに列を足し、`sim_iters` と `raw/` の表を足した。`runs.record_version = 2` |

| 段（spec の `record`、または `--record`） | 残すもの | 1 run（実測） |
|---|---|---|
| basic | `episodes`・`ticks`・`agents`・`sim_iters` | 0.23 MB |
| **standard（既定）** | ＋ `raw/agent_ticks`・`raw/pv_won` | 約 11 MB |
| detail | ＋ `raw/pv_all` | 約 92 MB |
| full | ＋ `raw/bids_all` | 約 481 MB |

- 段は run_id に入らない（結果は同じで、残す量だけが違う）。同じ run_id が低い段で既にあるとき、高い段を指定して流すと、流し直して置き換える。
- **リポジトリ全体で使ってよい容量は 200 GB まで**（ユーザーの決定、2026-10-10）。detail・full を多くの run で取る前に、`記録の一覧.md` の 1 節で見積もり、`du -sh .` で確かめる。
- `github/` を変えたので、同じ設定でも、版 2 の run_id は版 1 と違う。**同じ設定の run は、`runs.config_key`（コードの版を含まない、設定だけのハッシュ）と `player_index` で結ぶ。**
- 1 プロセスのメモリは、約 3.2 GB（版 1 は約 2.8 GB）。並列数は、これで見積もる。

## 使い方

```bash
# 実験を回す（spec.json は research/W001_.../REP002_.../params/ などに置く）
.venv/Scripts/python.exe research/src/run_experiment.py <spec.json> --workers 4
# DB を作り直す
.venv/Scripts/python.exe DB/build_db.py
# 問い合わせ
.venv/Scripts/python.exe DB/query.py "SELECT * FROM run_summary ORDER BY score DESC"
.venv/Scripts/python.exe DB/query.py --describe      # 表・列の意味を一覧
```

Windows で日本語が文字化けするときは `PYTHONUTF8=1` を付ける。`.venv` は `bash github/Test/1_初期セットアップ/setup_env.sh` で作り、`duckdb` と `pyarrow` を追加で入れる（`research/requirements.txt`）。

spec.json の書き方は `research/src/run_experiment.py` の冒頭を参照。変える値だけを書き、`sweep` でパラメータを振る。

### 学習ベースの手法（BC / IQL / CQL / BCQ / TD3_BC）

```bash
bash github/Test/2_学習データと学習手法/download_data.sh               # 公開データを DB/dataset/ に取得（約80GB）
.venv/bin/python research/src/make_train_data.py --workers 2     # 学習データと DB/train_data/ の要約を作る
.venv/bin/python research/src/train_model.py <spec.json>         # 学習。重みは DB/models/<model_id>/ に入る
.venv/bin/python research/src/run_experiment.py <spec.json>      # 評価。player.kwargs.model_dir に "DB/models/<model_id>" を書く
```

- 学習の spec の書き方は `research/src/train_model.py` の冒頭を参照（`algo`、`step_num`、`seed`、`sweep`）。
- **model_id は「手法・ステップ数・乱数シード・学習データの SHA-1・コードの版」のハッシュ。** 同じ条件なら同じ ID になり、既にあればスキップする。
  乱数（`random`・numpy・torch）を固定しているので、同じ条件の学習は同じ重み（同じ `model_sha1`）になる。
- 評価の spec で `model_dir` を書かなければ、シミュレータに同梱の重み（`official_agent/`）を使う。その run は `runs.model_id` が NULL。
- 学習も評価も CPU で行う（`CUDA_VISIBLE_DEVICES=""` を実行スクリプトが設定する）。
- 道具の検証は `github/Test/2_学習データと学習手法/`。

## 表

列の意味・単位・「なぜ取るか」は `columns.json`（元は `make_columns.py`）。DuckDB 上でも、表と列の COMMENT として読める（`query.py --describe`）。

| 表 | 粒度 | 中身 |
|---|---|---|
| `runs` | 1 実行 | 設定（params_spec / params_effective）、コミット、端末・版、所要時間、status、論文式のスコア |
| `episodes` | 実行 × エピソード | プレイヤーのエピソード成績（報酬、CPA、ペナルティ、予算消化率、48 人中の順位 など） |
| `ticks` | 実行 × エピソード × ティック | プレイヤー視点の時系列（実効入札係数 alpha、落札数、スロット別、支払い、残り予算、市場価格、競合の脱落数、bidding の所要時間 など） |
| `agents` | 実行 × エピソード × 広告主（48） | 背景の広告主を含む全員の成績 |
| `params_long` | 実行 × パラメータ | `params_effective` を縦に展開（値で run を絞る・結合する用） |
| `models` | 学習した重み | 手法、ステップ数、乱数シード、学習データの SHA-1、重みの SHA-1、損失の要約、所要時間、コミット。`runs.model_id` で結ぶ |
| `run_summary`（view） | 実行 | よく使う要約 |
| `sim_iters` | 実行 × エピソード × ティック × 環境の呼び出し | 予算超過のやり直しの各回（超過した広告主、プレイヤーの支払い・購入、時間）。版 2 だけ |
| `agent_ticks`（view） | 実行 × エピソード × ティック × 広告主（48） | 全員のティックごとの集計（入札額の分布、α、落札・露出・支払い・購入、残り予算、N_est・N_real）と、手法の内部状態。段 standard 以上 |
| `pv_won`（view） | プレイヤーが落札した機会 ＋ 取り消された入札 | 推定価値、σ、雑音を加えた値、入札額、枠、支払い単価、露出、購入の抽選、競合の上位。段 standard 以上 |
| `pv_all`（view） | プレイヤーの全機会 | 同上（落札しなかった機会を含む）。段 detail 以上 |
| `bids_all`（view） | 全機会 | 全 48 社の入札額（列 `bid_00`〜`bid_47`）、枠を得た広告主、露出した枠（ビット）。段 full。列の辞書には入れていない |
| `epl`（view） | 実行 × エピソード × 広告主 | REP006 の三つの指標 E・P・L と、R = E × P × L、抽選の z。版 2 だけ |
| `runs_newest`（view） | 同じ設定 × 位置 | `status = 'ok'`・`github_dirty = false` の、いちばん新しい run |

- `ticks`・`episodes`・`agents`・`runs` には、版 2 で列を足した（REP006 の N_est・N_real・Σp(1−p)、枠別の露出・支払い、入札額と市場価格の分位点、予算超過のやり直し、時間の内訳、資源、乱数の種、設定の全文 など）。版 1 の run では NULL。
- `raw/` の表は、DuckDB では、parquet を直接読む view である（コピーしない）。`raw/` が無い run は、入らない。機会ごとの表は float32 なので、正確な合計は `ticks`・`episodes`・`agents` を使う。

例：

```sql
-- 手法ごとの、位置・エピソードを通した平均スコア（score_component は 20000 で割る前）
SELECT r.strategy, avg(e.score_component) AS score_component_mean, count(*) AS n
FROM episodes e JOIN runs r USING (run_id) WHERE r.github_dirty = false GROUP BY 1;

-- 自前で学習した重みの成績を、学習の条件と並べる
SELECT m.algo, m.step_num, m.seed, avg(e.score_component) AS score_component_mean, count(*) AS n
FROM episodes e JOIN runs r USING (run_id) JOIN models m USING (model_id) GROUP BY 1, 2, 3;

-- REP006 の三つの指標（プレイヤー、購入のあるセル）。恒等式 R = E × P × L
SELECT strategy, run_id, episode, E, P, L, R, z FROM epl WHERE is_player AND reward >= 1;

-- 版 1 と版 2 の、同じ設定の run を並べる
SELECT a.run_id AS v1, b.run_id AS v2, a.strategy, a.player_index
FROM runs a JOIN runs b USING (config_key, player_index) WHERE a.record_version = 1 AND b.record_version = 2;

-- PID の base_action を振ったときの成績
SELECT p.value_num AS base_action, avg(e.score_component), avg(e.penalty)
FROM episodes e JOIN params_long p USING (run_id)
WHERE p.key = 'player.kwargs.base_action' GROUP BY 1 ORDER BY 1;
```

## 列を足すとき

0. `記録の一覧.md` に、何を、なぜ、どの段で取るかを足す。
1. `make_columns.py` に、意味・単位・「なぜ取るか」を書く。
2. `python DB/make_columns.py` → `columns.json` を作り直す。
3. `research/src/run_experiment.py` で値を記録する。
4. `python DB/build_db.py`。辞書にない列・実データにない列は、警告が出る。
5. `github/` を変えて量を取り出すときは、**結果を変えない**（乱数を引かない）。`github/Test/1_初期セットアップ/` を全件、`github/Test/3_実験記録の拡充/` を回し、既存の run と一致することを確かめる（`test_05_same_as_old_runs.py`）。

古い run には、新しい列がない（NULL になる）。原本は書き換えない。

## パラメータの扱い（JSON → 本家コード）

- `github/` で、コードに直書きだった値を、**既定値を従来のままにして**引数へ出した。`run_experiment.py` が `spec.json` から渡す。
  - PID：`base_action`（初期の入札係数 α）、`up_factor`、`down_factor`、`low_threshold`、`high_threshold`
  - ABid：`bid_scale`（α に掛ける倍率）
  - OnlineLP：`cpa_cap_ratio`（α の上限 = CPA × この値）、`episode`（どの日の早見表を使うか）
  - BC / IQL / CQL / BCQ / TD3_BC（プレイヤーとして使うとき）：`model_dir`（重みのフォルダ。既定は同梱の重み）
  - 学習（`run_*.py` の学習関数）：`train_data_path`、`save_path`、`step_num`。`train_model.py` が spec から渡す
  - 環境（gin）：`BiddingEnv.slot_coefficients` / `default_seed` / `conversion_seed`、`Controller.budget_ratio`、`PlayerAnalysis.penalty_beta` / `score_normalizer`、既存の `PVNUM`・`RESERVE_PV_PRICE` など
- `github/config/test.gin` を読んだあと、`spec.json` の `gin` を上書きとして適用する。
- 実際に使われた値は、`params.json` の `effective` と `params_long` に残る。

### まだ固定されている値（json から変えられない）

コードを読んで確認したもの（変えたい場面が出たら、引数に出す）。

- 背景の広告主 47 人の構成（PID・IQL・TD3_BC・OnlineLP・CQL・BC・MOPO・BCQ・COMBO の並び）と、その戦略のパラメータ・学習済みモデル
- 学習のハイパーパラメータ（学習率、ネットワークの大きさ、バッチ 100、IQL の expectile など）と、学習データの作り方（状態 16 次元、報酬 = 露出した PV の価値の和）
- 48 人の予算（`Controller.calculate_budget`）と CPA 制約（`get_cpa_constraints`）の表。`budget_ratio` で一律に倍率をかけることだけできる
- 広告機会の生成（`NeurIPSPvGen`）の定数：トラフィックの揺らぎ（scale 0.4, window 4）、価値の平均 0.0005 と揺らぎ、ばらつき・不確実性の分布、乱数シード（`episode` から決まる）。`episode` を変えることでだけ、広告機会が変わる
- `BiddingEnv`：スロット数 3、`MAGIC_NUMBER = 1019`、価値のノイズの打ち切り範囲（`±2 × 乱数`）
- 乱数：`BiddingEnv` の露出・価値ノイズ・コンバージョンの抽選は、呼ぶたびに同じシードで作り直される（`default_rng(seed=...)`）。**ティックごと・エピソードごとに抽選の乱数列が変わらない。** 2026-10-10 に、関数を直接呼んで確かめた（`github/Test/3_実験記録の拡充/test_02_env_facts.py`）：雑音（標準化したもの）は、（ティックの中の機会の番号、広告主）だけで決まる。購入の抽選は、確率が 0 の機会が乱数を使わないので、そこから先の当たり方がずれる。評価のばらつきは、エピソード番号（広告機会の違い）と位置の違いから来る。
- 過払い調整（`run_test.adjust_over_cost`）の乱数シード 1
- `run_test` が使う PID フォールバックの構成（本実験では使わず、`player_agent` を渡す）

## 注意

- `github/` を変更したら、`github/Test/1_初期セットアップ/` の pytest を回す（変更したパラメータの回帰は `test_17_params_injection.py`、`test_18_model_dir_and_train_args.py`）。
- **端末をまたぐと、結果は完全には一致しない。** Windows（REP002）と Linux で同じ 24 セルを比べると、23 セルは一致し、1 セル（ABid・位置 0・エピソード 0）で購入数が違った（REP003）。手法どうしを比べるときは、`runs.host` / `os` が同じ run だけを使う。
- 実験で使う結果を引くときは、`github_dirty = false`、`status = 'ok'` で絞る。
- 重い生データ（機会ごとの記録、全入札のログ）は、`runs/<run_id>/raw/` に置く（`.gitignore` 済み。記録の段 standard 以上で、自動で書かれる）。git に入らないので、サーバーが壊れると失われる。同じコードの版・同じ設定で流し直せば、同じものができる。
- 容量を測るだけの試し流しは、`run_experiment.py --runs-dir <一時の場所>` で、`DB/runs/` の外に出す。
