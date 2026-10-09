# dashboards/ — サーバー・エージェント・使用量の画面

3 つの画面を、1 つの配信プログラムで出す。上のタブで切り替える。

| 画面 | 中身 | 要件 |
|---|---|---|
| サーバー負荷（`server.html`） | CPU・メモリ・GPU・ディスクの現在値と推移 | [要件定義_1_サーバー負荷.md](要件定義_1_サーバー負荷.md) |
| AI エージェント（`agents.html`） | タスク（予約）の一覧と、動いている Claude の一覧 | [要件定義_2_エージェントと使用量.md](要件定義_2_エージェントと使用量.md) |
| Claude の使用量（`usage.html`） | トークンの使用量の推移と内訳 | 同上 |

## 見る

サーバーの上で、配信用のプログラムを動かしておく：

```bash
cd dashboards && nohup programs/serve.py > logs/serve.log 2>&1 &
```

ブラウザで `http://localhost:8765/` を開く。VSCode の Remote で入っているときは、「ポート」タブで 8765 を転送する（自動で転送されることが多い）。
止めるときは `pkill -f "programs/serve[.]py"`。ポートを変えるときは `programs/serve.py --port 番号`。

- **「更新」ボタン**を押すと、サーバー負荷を集め直し（約 3 秒）、3 つの画面を作り直す。定期的な自動更新はしない。
- 画面を使わずに 1 回動かすには `programs/run.sh`。
- `serve.py` は、既定ではサーバー自身（127.0.0.1）からの接続だけを受ける。

## Claude を無人で動かす（Wi-Fi が切れても止まらない）

サーバーの tmux の中で Claude を動かす。ssh が切れても、動き続ける。

```bash
# 1. タスク（予約）を作る。フォルダと番号を予約する
python3 dashboards/programs/agentctl.py new --title "REP006 の実験" --summary "何のためにやるか" \
    --paths "research/Q001_xxx/REP006_yyy" --ids REP006

# 2. 指示書を書く（やること＝チェックリスト、終わりの条件）
#    dashboards/agents/tasks/T001/指示書.md

# 3. 起動する（tmux の中で、確認なしで全部実行）
python3 dashboards/programs/agentctl.py launch T001

# 様子を見る（離れるのは Ctrl-b → d。Claude は動き続ける）
tmux attach -t agent-T001

# 一覧／止める／終わらせる
python3 dashboards/programs/agentctl.py list
python3 dashboards/programs/agentctl.py stop T001      # tmux を止めて「中止」
python3 dashboards/programs/agentctl.py done T001      # 「完了」（Claude が自分で実行する）
```

- 1〜3 は、VSCode の Claude に「このタスクを予約して、指示書を書いて、起動して」と頼めばよい。
- 起動した Claude は、`.claude/rules/unattended.md`（無人で動くときに守ること）と、指示書を読んでから始める。
- 確認なしの設定で初めて起動するとき、同意の画面が 1 度出ることがある。出たら `tmux attach` して同意する。
- 進み具合は、指示書のチェックリストの印の数で、画面に出る。

## 衝突を防ぐ仕組み

- **予約**：タスクごとに、フォルダ（`--paths`）と番号（`--ids`）を予約する。重なる予約は、`new` の時点で断られる（`--force` で重ねられる）。
- **警告**：他のタスクが予約している場所、またはホーム（`/home/yayoten`）の外を編集しようとした Claude には、その場で警告が出る。編集は止めない。警告は、画面の上にも出る。
- **自動の記録**：どの Claude が、どのファイルを編集したかは、フック（`.claude/settings.json`）が自動で記録する。VSCode の Claude も、tmux の Claude も対象。
- 限界：`Bash` の中での書き換え（`sed -i`、`mv` など）は、警告も記録も出ない。別の端末（自宅の PC など）で動かす Claude は、この台帳に出ない。

## ファイル

| ファイル | 内容 |
|---|---|
| `programs/serve.py` | 3 つの画面を配り、「更新」ボタンで `run.sh` を呼ぶ |
| `programs/run.sh` | 収集 → 3 つの画面の作り直し。ログは `logs/run.log` |
| `programs/collect.py` | サーバー負荷を 1 回分集めて `history/` に追記する |
| `programs/build_server.py` / `build_agents.py` / `build_usage.py` | それぞれの画面を作る |
| `programs/common.py` | 3 つの画面で共通の見た目（CSS）・タブ・書き出し |
| `programs/servers.json` | サーバー負荷の設定（対象、ディスク、しきい値、保持日数） |
| `programs/agentctl.py` | タスクの予約・起動・終了 |
| `programs/agent_hook.py` | フックから呼ばれ、動いている Claude を台帳に記録し、警告を返す |
| `programs/agents_lib.py` / `transcripts.py` | 台帳の読み書き／会話の記録の読み込み |
| `agents/` | 台帳。[agents/README.md](agents/README.md) |
| `history/`、`logs/`、`*.html` | 生成物（git では管理しない） |

## 数値の読み方（サーバー負荷）

- **CPU の混み具合**：15分ロードアベレージ ÷ コア数。100% で全コアが埋まっている目安。
- **CPU 使用率・GPU 使用率**：収集した瞬間の値。短い山や谷は拾えない。
- **推移グラフ**：「更新」を押した時刻の値しか無い。押していない間の負荷は分からない。
- **メモリ**：空きは「利用可能量（available）」。キャッシュは使用中に数えない。
- 設定は `programs/servers.json`。しきい値は `[混雑, 逼迫]` の順（メモリだけは「空きがこの割合を下回ったら」）。

## 止まったとき

`logs/run.log` を見る。1 回につき `collect` と、3 つの `build` の行が出る。
