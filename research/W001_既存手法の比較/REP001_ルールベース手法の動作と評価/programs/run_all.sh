#!/usr/bin/env bash
# REP001 の全実験を順に流し、DB を作り直す。既にある run はスキップされる（再実行しても安全）。
#   bash programs/run_all.sh [workers]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
export PYTHONUTF8=1
PY="$ROOT/.venv/Scripts/python.exe"; [ -x "$PY" ] || PY="$ROOT/.venv/bin/python"
W="${1:-8}"
for spec in base_all sweep_pid_base_action sweep_pid_up_factor sweep_pid_down_factor sweep_pid_low_threshold \
            sweep_pid_high_threshold sweep_abid_bid_scale sweep_olp_cpa_cap_ratio sweep_olp_table_episode; do
  echo "=== $spec $(date '+%F %T')"
  "$PY" "$ROOT/research/src/run_experiment.py" "$HERE/../params/$spec.json" --workers "$W" 2>&1 | grep -v "^INFO"
done
"$PY" "$ROOT/DB/build_db.py"
echo "=== done $(date '+%F %T')"
