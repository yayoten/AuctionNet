#!/usr/bin/env bash
# 学習用の公開データ（period-7〜27、zip 11本で約18GB、展開後 約80GB）を取得して展開する。
#
#   bash github/Test/2_学習データと学習手法/download_data.sh            # リポジトリ直下の DB/dataset/ に置く
#   bash github/Test/2_学習データと学習手法/download_data.sh 置き場所DIR
#
# - URL の出どころは github/pre_generated_dataset/readme_dataset.md。
# - 途中で止まっても、再実行すれば続きから取得する（aria2c -c / wget -c）。展開済みの zip は飛ばす。
# - 展開先は <置き場所>/traffic/period-N.csv。zip は <置き場所>/zip/ に残す（消してよい）。
# - DB/dataset/ は git 管理外（.gitignore）。名前を data/ にしないのは、github/Test/1 の test_16 が
#   「リポジトリ直下に data/traffic が無い」ことを前提にしているため。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
DEST="${1:-$ROOT/DB/dataset}"
BASE="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final"
PARTS=(7-8 9-10 11-12 13 14-15 16-17 18-19 20-21 22-23 24-25 26-27)

mkdir -p "$DEST/zip" "$DEST/traffic"
for p in "${PARTS[@]}"; do
  f="autoBidding_general_track_final_data_period_$p.zip"
  if [ -f "$DEST/zip/$f.done" ]; then echo "== 済: $f"; continue; fi
  echo "== 取得: $f $(date '+%F %T')"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 8 -s 8 --console-log-level=warn --summary-interval=0 -d "$DEST/zip" -o "$f" "$BASE/$f"
  else
    wget -c -q -O "$DEST/zip/$f" "$BASE/$f"
  fi
  unzip -tq "$DEST/zip/$f" >/dev/null          # 壊れていればここで止まる
  unzip -oq -j "$DEST/zip/$f" '*.csv' -x '*/._*' '._*' -d "$DEST/traffic"   # macOS の ._*.csv は展開しない
  touch "$DEST/zip/$f.done"
done
echo "== 完了 $(date '+%F %T')"
ls -la "$DEST/traffic" | grep period- || true
