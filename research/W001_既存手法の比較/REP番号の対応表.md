# REP 番号の対応表（W001）

2026-10-10 に、ユーザーの指示で、W001 の REP の番号を付け替えた。**「ID は再利用しない」（`research/README.md` 4.）の例外**である（2 回目。1 回目は 2026-10-08、REP003 ← 題未定の要件定義）。

## 対応

| 新 | 題 | 旧 | 備考 |
|---|---|---|---|
| REP001 | ベースラインのパラメータ設定 | REP003 | 以後の比較のベースラインの設定を決めた。成果物は `research/src/baseline_params.json` |
| REP002 | ルールベース手法 3 種の動作検証（Windows、掃引つき） | REP001 | |
| REP003 | 学習ベース手法 5 種の動作検証（既定設定で学習し直し） | REP002 | |
| REP004 | （欠番） | REP004 | 要件定義の案だけがあった。内容は `Q001/chat_with_claude/0_…` へ移した |
| REP005 | 手法別の CPA 制約の超過（Q001） | REP005 | 動かしていない |
| REP006 | 超過の三つの指標への分解（Q001） | REP006 | 動かしていない |
| （番号未定） | `再実施_ルールベース/`（確定した設定でのルールベース 3 手法の検証） | （新しい REP） | タスク T003 の成果。完了 |
| （番号未定） | `再実施_学習ベース/`（確定した設定での学習ベース 5 手法の検証） | （新しい REP） | タスク T004 の成果。評価 720 run のうち 661 run で中止（2026-10-10） |

## 何を直し、何を直さなかったか

| 直した | 直さなかった（当時の番号のまま） |
|---|---|
| フォルダ名、REP の md・tex・PDF の ID、W001 の REP 内の相互参照 | **DB の `rep` の値**（`DB/runs/*/meta.json`、`episodes` などの `rep` 列。`REP001`〜`REP005`、`REP002new`、`REP003new`） |
| `W001.md`、`MASTER.md`（現在地、分かったこと）、`Q001.md`、REP005・REP006 の md | `MASTER.md` の「判断の履歴」の、2026-10-09 以前の行 |
| `research/src/baseline_params.json`（`decided_in`、spec のパス）、`make_baseline_json.py`、`test_06_baseline_params.py` | 過去のコミットメッセージ |
| `github/Test/` の README、`DB/README.md` の例 | `chat_with_claude/` の資料（番号は書いた日の番号） |
| 問いのまとめ PDF、`flow.md` | `dashboards/` の台帳・指示書・経緯・進捗（タスクの記録） |
| | 各 REP の `programs/` と `params/` の中の、DB の `rep` を絞り込む値（旧番号のままでないと、DB から run を引けない） |

## 読み方の注意

- **DB の `rep` が `REP001` の run は、新 REP002（Windows のルールベース）のもの。** `REP002` の run は新 REP003（学習ベースの既定設定）、`REP003` の run は新 REP001（ベースラインの設定を決めるための run）のもの。
- `rep` が `REP002new`、`REP003new` の run は、`再実施_ルールベース/`、`再実施_学習ベース/` の run。
- `chat_with_claude/` や `dashboards/` で「REP001」「REP002」「REP003」と出てきたら、この表で読み替える（旧番号）。
