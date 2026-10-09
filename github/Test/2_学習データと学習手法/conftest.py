"""学習データの取得から、学習、シミュレータでの評価までを、段階ごとに確かめるテストの共通設定。

方針
- 段階の順にファイルを並べる（01 原本 → 02 学習データ → 03 学習 → 04 評価 → 05 本番の重み）。前の段階が通ってから次へ進む。
- 80GB の原本（DB/dataset/）が要るテストは `needs_data`。原本が無い端末では、DB/train_data/ の要約で確かめられるものだけ走る。
- テストは DB/ にも github/ にも書かない。学習・評価の出力は tmp に出す。
- CPU で動かす（ロックの torch 1.12.0 は、新しい GPU では CUDA の計算が落ちる）。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
GITHUB_DIR = REPO_ROOT / "github"
DATASET = REPO_ROOT / "DB" / "dataset" / "traffic"
RL_DIR = DATASET / "training_data_rlData_folder"
SUMMARY_DIR = REPO_ROOT / "DB" / "train_data"
MODELS_DIR = REPO_ROOT / "DB" / "models"
PERIODS = list(range(7, 28))  # 公開データは period-7 〜 period-27 の 21 日分

for p in (str(REPO_ROOT), str(REPO_ROOT / "research" / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gin  # noqa: E402
import github.run.run_test  # noqa: E402,F401  （本家が import 時に gin へ登録する）


def _github_status():
    import subprocess
    return subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "github", "DB"], cwd=REPO_ROOT,
                          capture_output=True, text=True).stdout


STATUS_AT_START = _github_status()


def pytest_collection_modifyitems(config, items):
    if DATASET.is_dir() and any(DATASET.glob("period-*.csv")):
        return
    skip = pytest.mark.skip(reason="DB/dataset/traffic が無い。bash github/Test/2_学習データと学習手法/download_data.sh で取得する")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()


@pytest.fixture(scope="session")
def rl_csv(tmp_path_factory):
    """学習データ（rlData）と同じ列・同じ書式の、小さな合成データ。6 本の軌跡 × 48 ティック。"""
    rng = np.random.default_rng(0)
    rows = []
    for traj in range(6):
        states = [tuple(float(x) for x in np.r_[(48 - t) / 48, 1 - t / 60, rng.uniform(0, 1, 11), rng.integers(100, 5000, 3)])
                  for t in range(48)]
        for t in range(48):
            rows.append(dict(deliveryPeriodIndex=7, advertiserNumber=traj, advertiserCategoryIndex=0, budget=3000.0,
                             CPAConstraint=100.0, realAllCost=2000.0, realAllConversion=20.0, timeStepIndex=t,
                             state=states[t], action=float(rng.uniform(20, 150)), reward=float(rng.integers(0, 3)),
                             reward_continuous=float(rng.uniform(0, 2)), done=int(t == 47),
                             next_state=states[t + 1] if t < 47 else None))
    path = tmp_path_factory.mktemp("rl") / "training_data_all-rlData.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
