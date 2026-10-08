# サーバー負荷ダッシュボード

サーバーの CPU・メモリ・GPU・ディスクの負荷を15分ごとに集め、`dashboard.html` に一覧と推移を出す。
要件は [要件定義.md](要件定義.md)。

## 見る

`server_info/dashboard.html` を開く（VSCode なら右クリック →「Open Preview」、または手元にダウンロードしてブラウザで開く）。
自動では再読み込みしないので、開き直すと最新になる。最後の更新から30分以上たっていると、画面の上に警告が出る。

## 手で1回動かす

```bash
./run.sh
```

履歴に1回分が追記され、`dashboard.html` が作り直される。

## 15分ごとに動かす（cron）

登録：

```bash
( crontab -l 2>/dev/null; echo "*/15 * * * * /home/yayoten/AuctionNet/server_info/run.sh" ) | crontab -
```

確認は `crontab -l`。解除は `crontab -e` で該当の行を消す。

## ファイル

| ファイル | 内容 |
|---|---|
| `servers.json` | 対象サーバー、監視するディスク、色分けのしきい値、履歴の保持日数 |
| `collect.py` | 1回分を集めて `history/` に追記する。保持日数を過ぎた行はここで消す |
| `build_dashboard.py` | `history/` から `dashboard.html` を作る |
| `run.sh` | cron から呼ぶ。収集 → 画面の作り直し。ログは `logs/run.log` |
| `history/host.csv` | CPU・メモリ（1回につきサーバーごとに1行） |
| `history/gpu.csv` | GPU（1回につき GPU ごとに1行） |
| `history/disk.csv` | ディスク（1回につきマウントごとに1行） |
| `history/latest.json` | 最新の回の GPU 使用プロセスと、取得に失敗した項目 |

`dashboard.html`、`history/`、`logs/` は生成物で、git では管理しない。

## 設定を変える

`servers.json` を書き換える。次の回から反映される。

- 監視するディスクを増やす：`disks` にマウント先を足す。
- しきい値：`thresholds` の各項目は `[混雑, 逼迫]` の順。メモリだけは「空きがこの割合を下回ったら」。

## 数値の読み方

- **CPU の混み具合**：15分ロードアベレージ ÷ コア数。100% で全コアが埋まっている目安。15分に1回の収集でも、その間の平均が分かる。
- **CPU 使用率・GPU 使用率**：収集した瞬間の値。短い山や谷は拾えない。
- **メモリ**：空きは「利用可能量（available）」。キャッシュは使用中に数えない。
- **ディスク使用率**：`df` と同じ計算（使用 ÷（使用＋一般ユーザーが使える空き））。

## 止まったとき

`logs/run.log` を見る。1回につき `collect` と `build` の2行が出る。取得に失敗した項目は、`collect` の行の `errors=` と、画面のサーバー名の下に出る。
