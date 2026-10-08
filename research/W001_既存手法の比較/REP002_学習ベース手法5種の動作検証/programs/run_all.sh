#!/usr/bin/env bash
# REP002 の全実験を順に流す。既にある学習・run はスキップされる（再実行しても安全）。
#   bash programs/run_all.sh [評価の並列数=8] [学習の並列数=4]
# 前提：公開データと学習データがある（Test/2_学習データと学習手法/README.md の手順 1, 2）。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY="$ROOT/.venv/Scripts/python.exe"
export PYTHONUTF8=1
W="${1:-8}"; TW="${2:-4}"
P="$HERE/../params"
for spec in train_default train_seeds train_steps; do
  [ -f "$P/$spec.json" ] || continue
  echo "=== 学習 $spec $(date '+%F %T')"
  "$PY" "$ROOT/research/src/train_model.py" "$P/$spec.json" --workers "$TW" 2>&1 | grep -v -E "^INFO|UserWarning|pred_actions"
done
"$PY" "$HERE/make_eval_specs.py"
for spec in "$P"/eval_trained_default_*.json "$P"/eval_rule_based.json "$P"/eval_bundled.json "$P"/eval_trained_steps_*.json "$P"/eval_trained_seeds_*.json; do
  [ -f "$spec" ] || continue
  echo "=== 評価 $(basename "$spec") $(date '+%F %T')"
  "$PY" "$ROOT/research/src/run_experiment.py" "$spec" --workers "$W" 2>&1 | grep -v "^INFO"
done
"$PY" "$ROOT/DB/build_db.py"
echo "=== done $(date '+%F %T')"
