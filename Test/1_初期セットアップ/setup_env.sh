#!/usr/bin/env bash
# 検証用の Python 3.9 仮想環境を作る（macOS / Linux / Windows の Git Bash・WSL）。
#
#   bash Test/1_初期セットアップ/setup_env.sh              # リポジトリ直下の .venv に作る
#   bash Test/1_初期セットアップ/setup_env.sh --no-lock    # ロックを使わず github/requirements.txt から解決
#   bash Test/1_初期セットアップ/setup_env.sh --pip        # uv があっても使わず、python3.9 + pip で作る
#   bash Test/1_初期セットアップ/setup_env.sh 作成先DIR    # 作成先を変える
#
# - uv があれば uv を使う（Python 3.9 が無くても uv が取得する）。無ければ、手元の Python 3.9 + pip で作る。
# - 既定では requirements.lock.txt（全パッケージのバージョン固定）を入れる。どの端末でも同じ版になる。
# - torch==1.12.0 は Python 3.10 以降に配布が無い。Python は 3.9 を使う。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VENV="$ROOT/.venv"
USE_LOCK=1
FORCE_PIP=0
for arg in "$@"; do
  case "$arg" in
    --no-lock) USE_LOCK=0 ;;
    --pip) FORCE_PIP=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    -*) echo "不明なオプション: $arg" >&2; exit 2 ;;
    *) VENV="$arg" ;;
  esac
done

if [ "$USE_LOCK" = 1 ]; then
  REQ_ARGS=(-r "$HERE/requirements.lock.txt")
else
  REQ_ARGS=(-r "$ROOT/github/requirements.txt" pytest pytest-timeout)
fi

venv_python() {  # Windows は Scripts/、それ以外は bin/
  if [ -x "$VENV/bin/python" ]; then echo "$VENV/bin/python"; else echo "$VENV/Scripts/python.exe"; fi
}

find_python39() {
  local c
  for c in python3.9 python3 python "py -3.9" /usr/bin/python3; do
    if $c -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 9) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

if [ "$FORCE_PIP" = 0 ] && command -v uv >/dev/null 2>&1; then
  echo "== uv で作成: $VENV"
  uv venv --python 3.9 "$VENV"
  uv pip install --python "$(venv_python)" "${REQ_ARGS[@]}"
else
  PY="$(find_python39)" || {
    echo "Python 3.9 が見つかりません。次のどちらかを用意してください:" >&2
    echo "  - uv を入れる（Python 3.9 も uv が取得します）: https://docs.astral.sh/uv/" >&2
    echo "  - Python 3.9 を入れる（pyenv install 3.9 / conda create -n AuctionNet python=3.9 など）" >&2
    exit 1
  }
  echo "== $PY + pip で作成: $VENV"
  $PY -m venv "$VENV"
  "$(venv_python)" -m pip install --quiet --upgrade pip
  "$(venv_python)" -m pip install --quiet "${REQ_ARGS[@]}"
fi

PYBIN="$(venv_python)"
"$PYBIN" - <<'PY'
import platform, sys
import einops, gin, matplotlib, numpy, pandas, scipy, torch
print("python", sys.version.split()[0], "|", platform.system(), platform.machine())
print("torch", torch.__version__, "| numpy", numpy.__version__, "| pandas", pandas.__version__,
      "| scipy", scipy.__version__, "| cuda", torch.cuda.is_available())
assert sys.version_info[:2] == (3, 9), "Python 3.9 ではありません"
assert hasattr(gin, "configurable")
PY
echo "OK: $VENV"
echo "テスト: cd \"$HERE\" && \"$PYBIN\" -m pytest"
echo "本体  : cd \"$ROOT/github\" && \"$PYBIN\" main_test.py"
