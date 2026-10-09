#!/bin/bash
# serve.py（画面の「更新」ボタン）から呼ぶ。手でも動かせる。サーバー負荷の収集 → 3 つの画面の作り直し。
# 収集に失敗しても、ダッシュボードは作り直す（取れた分と、失敗した旨を画面に出すため）。
set -u
export PATH=/usr/local/bin:/usr/bin:/bin
cd "$(dirname "$0")" || exit 1
ROOT=..
mkdir -p "$ROOT/logs" "$ROOT/history"
LOG="$ROOT/logs/run.log"

# 前回がまだ動いていたら、今回は見送る
exec 9> "$ROOT/history/.lock"
if ! flock -n 9; then
    echo "$(date -Is) skip: 前回の実行が終わっていません" >> "$LOG"
    exit 0
fi

status=0
python3 collect.py >> "$LOG" 2>&1 || status=$?
python3 build_server.py >> "$LOG" 2>&1 || status=$?
python3 build_agents.py >> "$LOG" 2>&1 || status=$?
python3 build_usage.py >> "$LOG" 2>&1 || status=$?

# ログは直近2000行だけ残す
tail -n 2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
exit $status
