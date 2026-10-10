#!/usr/bin/env bash
# 評価の spec を、順に流す。既に ok の run はスキップされる（再実行しても安全）。
#   bash programs/run_specs.sh <並列数> <spec の名前（.json なし）> ...
# 2 本目のランナーを後から足せるよう、spec ごとにロック（results/raw/locks/<名前>.lock）を取る。
# 他のランナーが取った spec は飛ばす。終わった spec には <名前>.done を置く。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
PY="$ROOT/.venv/bin/python"
export PYTHONUTF8=1
W="$1"; shift
P="$HERE/../params"; L="$HERE/../results/raw/locks"; mkdir -p "$L"
cd "$ROOT"
for name in "$@"; do
  [ -f "$P/$name.json" ] || { echo "spec が無い: $name"; continue; }
  mkdir "$L/$name.lock" 2>/dev/null || { echo "=== 飛ばす（他のランナーが担当）$name"; continue; }
  echo "=== 評価 $name workers=$W $(date '+%F %T') avail=$(free -g | awk '/Mem/{print $7}')GB"
  if [ "$name" = "eval_bundled_mbrl" ]; then
    "$PY" "$HERE/run_bundled_mbrl.py" "$P/$name.json" --workers "$W" 2>&1 | grep -v "^INFO"
  else
    "$PY" "$ROOT/research/src/run_experiment.py" "$P/$name.json" --workers "$W" 2>&1 | grep -v "^INFO"
  fi
  touch "$L/$name.done"
done
echo "=== done $(date '+%F %T')"
