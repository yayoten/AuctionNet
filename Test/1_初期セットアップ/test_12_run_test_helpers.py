"""12: run/run_test.py の補助関数（勝者判定・過払い調整・プレイヤー初期化）。"""
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import GITHUB_DIR, REPO_ROOT
import github.run.run_test as rt
from github.simul_bidding_env.Tracker.PlayerAnalysis import PlayerAnalysis


# ---------- get_winner ----------
def test_get_winner_hand_example():
    # 3 エージェント × 2 PV。PV0: a0=slot1,a1=slot2,a2=slot3 / PV1: a2=slot1 のみ
    slot_pit = np.array([[1, 0], [2, 0], [3, 1]])
    w = rt.get_winner(slot_pit)
    assert w.shape == (2, 3)
    assert w[0].tolist() == [0, 1, 2]
    assert w[1].tolist() == [2, -1, -1]


def test_get_winner_all_unsold():
    assert (rt.get_winner(np.zeros((5, 4), dtype=int)) == -1).all()


def test_get_winner_dtype_is_int():
    assert rt.get_winner(np.array([[1], [0]])).dtype.kind == "i"


@pytest.mark.parametrize("seed", range(5))
def test_get_winner_inverts_slot_assignment(seed):
    rng = np.random.default_rng(seed)
    A, N = 10, 50
    slot = np.zeros((A, N), dtype=int)
    for j in range(N):
        k = rng.integers(0, 4)
        winners = rng.choice(A, k, replace=False)
        slot[winners, j] = np.arange(1, k + 1)
    w = rt.get_winner(slot)
    for j in range(N):
        for s in range(3):
            if w[j, s] >= 0:
                assert slot[w[j, s], j] == s + 1
            else:
                assert not (slot[:, j] == s + 1).any()


# ---------- adjust_over_cost ----------
def make_winner(n_pv, agent=0, slot=0, n_agents=3):
    w = np.full((n_pv, 3), -1, dtype=int)
    w[:, slot] = agent
    return w


def test_adjust_over_cost_drops_ceil_fraction_of_bids():
    bids = np.ones((10, 3))
    ratio = np.array([0.25, 0.0, 0.0])
    rt.adjust_over_cost(bids, ratio, np.array([1, .8, .6]), make_winner(10))
    assert (bids[:, 0] == 0).sum() == 3  # ceil(10*0.25)=3
    assert (bids[:, 1:] == 1).all()


def test_adjust_over_cost_ratio_one_drops_all():
    bids = np.ones((10, 3))
    rt.adjust_over_cost(bids, np.array([1.0, 0, 0]), np.array([1, .8, .6]), make_winner(10))
    assert (bids[:, 0] == 0).all()


def test_adjust_over_cost_no_overcost_changes_nothing():
    bids = np.random.default_rng(0).random((10, 3))
    ref = bids.copy()
    rt.adjust_over_cost(bids, np.zeros(3), np.array([1, .8, .6]), make_winner(10))
    assert np.array_equal(bids, ref)


def test_adjust_over_cost_only_touches_winning_pvs_of_that_agent():
    w = np.full((10, 3), -1, dtype=int)
    w[:4, 0] = 0  # agent 0 は PV0-3 の slot1 を取っている
    bids = np.ones((10, 3))
    rt.adjust_over_cost(bids, np.array([1.0, 0, 0]), np.array([1, .8, .6]), w)
    assert (bids[:4, 0] == 0).all() and (bids[4:, 0] == 1).all()


def test_adjust_over_cost_is_deterministic():
    a, b = np.ones((50, 3)), np.ones((50, 3))
    for x in (a, b):
        rt.adjust_over_cost(x, np.array([0.3, 0, 0]), np.array([1, .8, .6]), make_winner(50))
    assert np.array_equal(a, b)


def test_adjust_over_cost_drops_per_slot_separately():
    w = np.full((10, 3), -1, dtype=int)
    w[:5, 0] = 0
    w[5:, 1] = 0
    bids = np.ones((10, 3))
    rt.adjust_over_cost(bids, np.array([0.2, 0, 0]), np.array([1, .8, .6]), w)
    assert (bids[:5, 0] == 0).sum() == 1 and (bids[5:, 0] == 0).sum() == 1  # それぞれ ceil(5*0.2)=1


def test_adjust_over_cost_multiple_agents():
    w = np.full((10, 3), -1, dtype=int)
    w[:5, 0] = 0
    w[5:, 0] = 1
    bids = np.ones((10, 3))
    rt.adjust_over_cost(bids, np.array([0.2, 0.4, 0]), np.array([1, .8, .6]), w)
    assert (bids[:, 0] == 0).sum() == 1 and (bids[:, 1] == 0).sum() == 2


# ---------- その他の補助 ----------
def test_initialize_player_analysis_type():
    assert isinstance(rt.initialize_player_analysis(), PlayerAnalysis)


def test_log_memory_usage_does_not_crash(caplog):
    caplog.set_level("INFO")
    rt.log_memory_usage()
    assert any("Memory usage" in r.message for r in caplog.records)


# ---------- initialize_player_agent ----------
def test_initialize_player_agent_falls_back_to_pid_when_training_package_missing():
    """strategy_train_env が sys.path に無ければ（=bidding_train_env を import できなければ）PID にフォールバック。"""
    assert "bidding_train_env" not in sys.modules
    a = rt.initialize_player_agent()
    assert type(a).__name__ == "PidBiddingStrategy" and a.name == "PidBiddingStrategy0"
    assert a.exp_budget_ratio.shape == (48,) and (a.exp_budget_ratio == 1).all()


def test_initialize_player_agent_fallback_is_a_fresh_instance_each_time():
    assert rt.initialize_player_agent() is not rt.initialize_player_agent()


def test_initialize_player_agent_falls_back_when_training_package_importable_but_no_model():
    """main_test.py は strategy_train_env を sys.path に足す。すると bidding_train_env は import できるが、
    クローン直後は学習済み IQL（saved_model）が無い。修正前はここで sys.exit(1)。今は PID にフォールバックする。"""
    code = (
        "import sys; sys.path.append('./strategy_train_env')\n"
        "import github.run.run_test as rt\n"
        "a = rt.initialize_player_agent()\n"
        "print('AGENT', type(a).__name__, a.name)\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=GITHUB_DIR, env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    assert "AGENT PidBiddingStrategy PidBiddingStrategy0" in r.stdout
    assert "saved_model/IQLtest/iql_model.pth" in r.stderr  # 理由はログに出る


def test_initialize_player_agent_never_calls_sys_exit():
    import inspect

    assert "sys.exit" not in inspect.getsource(rt.initialize_player_agent)


def test_initialize_player_agent_with_import_error_logs_and_falls_back(monkeypatch, caplog):
    import builtins

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "bidding_train_env.strategy":
            raise ImportError("boom")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    caplog.set_level("ERROR")
    a = rt.initialize_player_agent()
    assert type(a).__name__ == "PidBiddingStrategy"
    assert any("Failed to load PlayerAgent" in r.message for r in caplog.records)


def test_run_test_module_does_not_use_removed_collections_alias():
    """修正前は `from collections import Iterable`（Python 3.10 で削除）だった。"""
    src = (GITHUB_DIR / "run" / "run_test.py").read_text(encoding="utf-8")
    assert "from collections import Iterable" not in src
    assert "from collections.abc import Iterable" in src


def test_no_module_imports_removed_collections_aliases():
    import re as _re

    bad = [str(f) for f in GITHUB_DIR.rglob("*.py")
           if _re.search(r"^from collections import .*\b(Iterable|Mapping|Sequence|Callable)\b",
                         f.read_text(encoding="utf-8"), _re.M)]
    assert bad == []


def test_run_test_is_gin_configurable_with_documented_defaults():
    import inspect

    sig = inspect.signature(rt.run_test)
    # gin.configurable でラップされていても、元の既定値が見える
    assert "generate_log" in sig.parameters or "kwargs" in sig.parameters
