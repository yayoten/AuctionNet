"""実験記録の拡充（T006、記録の形式の版 2）を確かめるテストの共通設定。

方針
- 記録を足しても、結果（購入・支払い・成績）が変わらないことを、最初に確かめる。
- テストは DB/ にも github/ にも書かない。実行の出力は tmp に出す（run_experiment の DB_RUNS、build_db の RUNS・DB を差し替える）。
- CPU で動かす。縮小設定（PV 数 3000〜6000）で流す。
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
GITHUB_DIR = REPO_ROOT / "github"
DB_DIR = REPO_ROOT / "DB"

for p in (str(REPO_ROOT), str(REPO_ROOT / "research" / "src"), str(DB_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gin  # noqa: E402
import github.run.run_test  # noqa: E402,F401  （本家が import 時に gin へ登録する）
import run_experiment as rx  # noqa: E402


def _status():
    import subprocess
    return subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "github", "DB"], cwd=REPO_ROOT,
                          capture_output=True, text=True).stdout


STATUS_AT_START = _status()
PV = 6000


@pytest.fixture(autouse=True)
def _clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()
    os.chdir(HERE)          # run_experiment は github/ へ chdir する


def run_one(root, strategy="PID", record="standard", player_index=3, episodes=(0, 1), kwargs=None, pv=PV, name="t"):
    """縮小設定で 1 run 流す。出力先は root/runs。meta と、run のフォルダを返す。"""
    runs = Path(root) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    spec = dict(name=name, player=dict(strategy=strategy, kwargs=kwargs or {}), player_indices=[player_index],
                episodes=list(episodes), gin={"PVNUM": pv}, record=record)
    sp = Path(root) / f"spec_{strategy}_{record}_{player_index}.json"
    sp.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rx, "DB_RUNS", runs)
        tasks, _ = rx.build_tasks(str(sp), True)
        assert len(tasks) == 1
        m = rx.execute_run(tasks[0])
    assert m["status"] == "ok", m.get("traceback")
    return m, runs / m["run_id"]


@pytest.fixture(scope="session")
def session_root(tmp_path_factory):
    return tmp_path_factory.mktemp("t006")


@pytest.fixture(scope="session")
def full_runs(session_root):
    """4 手法（PID・ABid・Online LP・IQL）を、記録の段 full で流したもの。{手法: (meta, フォルダ)}"""
    return {s: run_one(session_root / "full", s, "full") for s in ("PID", "ABid", "OnlineLP", "IQL")}
