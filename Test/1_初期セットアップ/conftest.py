"""初期セットアップ検証の共通設定。

方針
- 本家コード（github/）は、テストからは変更しない。import して動かすだけ（github/ 自体は変更してよい方針）。
- 本家は `github.` を頭につけて import しているため、リポジトリのルートを sys.path に入れる。
- 実験の生成物（CSV・モデル）は、すべて tmp_path に出す。github/ 配下を汚さない。
- クローン直後にあった不具合は github/ 側で修正済み。該当テストは「修正後の正しい挙動」を検証する回帰テストにしてある。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
GITHUB_DIR = REPO_ROOT / "github"
SIM_DIR = GITHUB_DIR / "simul_bidding_env"
TRAIN_DIR = GITHUB_DIR / "strategy_train_env"

# `github.simul_bidding_env...` を import できるようにする
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gin  # noqa: E402

# 本家が import 時に gin へ登録するため、先に全部 import しておく
import github.run.run_test  # noqa: E402,F401
import github.simul_bidding_env.Controller.Controller  # noqa: E402,F401
import github.simul_bidding_env.Environment.BiddingEnv  # noqa: E402,F401

from helpers import apply_gin  # noqa: E402


def _github_status():
    import subprocess

    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "github"], cwd=REPO_ROOT,
                       capture_output=True, text=True)
    return r.stdout


GITHUB_STATUS_AT_START = _github_status()


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def github_dir():
    return GITHUB_DIR


@pytest.fixture(autouse=True)
def _clean_gin():
    """テスト間で gin の束縛が漏れないようにする。"""
    gin.clear_config()
    yield
    gin.clear_config()


@pytest.fixture(autouse=True)
def _isolate_training_package_path():
    """本家の main_*.py / main_test.py は import するだけで sys.path に strategy_train_env を足す（副作用）。
    すると `bidding_train_env` が import できてしまい、テストの前提（どの戦略がプレイヤーになるか）が
    実行順で変わってしまう。テスト間で漏れないよう、毎回取り除く。"""
    def clean():
        sys.path[:] = [p for p in sys.path if Path(p).name != "strategy_train_env"]
        for name in [n for n in sys.modules if n == "bidding_train_env" or n.startswith("bidding_train_env.")]:
            del sys.modules[name]

    clean()
    yield
    clean()


@pytest.fixture(autouse=True)
def _fixed_seed():
    """本家 main_test.py と同じ乱数シード。"""
    import torch

    torch.manual_seed(1)
    np.random.seed(1)


@pytest.fixture
def bind_small_gin():
    """縮小設定の gin 束縛を行う関数を返す。"""

    def _bind(pv_num=3000, num_episode=1, num_tick=48, generate_log=False):
        apply_gin(pv_num=pv_num, num_episode=num_episode, num_tick=num_tick, generate_log=generate_log)

    return _bind


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
