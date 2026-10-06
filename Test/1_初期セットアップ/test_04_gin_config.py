"""04: gin 設定（config/test.gin）が読み込め、値がコードに届くか。

修正前は、test.gin が `run.run_test` を import して二重登録（ambiguous selector）になり読めなかった。
`github.` 付きの import に直し、PVNUM も Controller.pv_num に束縛した。
"""
import os
import re
import subprocess
import sys

import gin
import pytest

from conftest import GITHUB_DIR, REPO_ROOT
from helpers import CTRL, ENV, RUN, apply_gin, gin_bindings

GIN_FILE = GITHUB_DIR / "config" / "test.gin"


def _gin_values():
    vals = {}
    for line in GIN_FILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z_]+)\s*=\s*([0-9.]+)\s*(#.*)?$", line.strip())
        if m:
            vals[m.group(1)] = float(m.group(2)) if "." in m.group(2) else int(m.group(2))
    return vals


VALS = _gin_values()


# ---------- test.gin の中身 ----------
@pytest.mark.parametrize("key,expected", [
    ("RESERVE_PV_PRICE", 0.0001), ("MIN_REMAINING_BUDGET", 0.1), ("PVNUM", 500000),
    ("NUM_CATERORY", 6), ("NUM_AGENT_CATERORY", 8), ("NUM_AGENT", 48),
    ("NUM_EPISODE", 2), ("NUM_TICK", 48),
])
def test_gin_macro_value(key, expected):
    assert VALS[key] == expected


def test_gin_agent_count_is_consistent():
    assert VALS["NUM_CATERORY"] * VALS["NUM_AGENT_CATERORY"] == VALS["NUM_AGENT"] == 48


def test_gin_generate_log_off_by_default():
    assert re.search(r"^GENERATE_LOG\s*=\s*False", GIN_FILE.read_text(), re.M)


def test_gin_pv_generator_type_is_neurips():
    assert 'Controller.pv_generator_type = "neuripsPvGen"' in GIN_FILE.read_text()


def test_gin_binds_every_declared_parameter_once():
    text = GIN_FILE.read_text()
    for target in ("run_test.generate_log", "run_test.num_episode", "run_test.num_tick",
                   "BiddingEnv.reserve_pv_price", "BiddingEnv.min_remaining_budget",
                   "Controller.num_agent_category", "Controller.num_category", "Controller.num_tick",
                   "Controller.pv_num", "Controller.pv_generator_type"):
        assert len(re.findall(rf"^{re.escape(target)}\s*=", text, re.M)) == 1, target


def test_gin_pvnum_macro_is_bound_to_controller():
    text = GIN_FILE.read_text()
    assert "PVNUM = 500000" in text
    assert "Controller.pv_num = %PVNUM" in text


def test_gin_imports_use_github_prefix():
    text = GIN_FILE.read_text()
    assert "import github.run.run_test" in text
    assert not re.search(r"^import (run|simul_bidding_env)\.", text, re.M)


def test_gin_num_agent_macro_is_defined_but_never_used():
    assert "%NUM_AGENT\n" not in GIN_FILE.read_text() + "\n"


# ---------- config/test.gin そのもの ----------
def q(key):
    """gin はマクロ(%NAME)で束縛した値を参照のまま返すので、実値まで解決する。"""
    v = gin.query_parameter(key)
    return gin.query_parameter(str(v)) if str(v).startswith("%") else v


def _run_in_github(code: str, **kw):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=GITHUB_DIR, env=env,
                          capture_output=True, text=True, timeout=300, **kw)


def test_test_gin_parses_cleanly_in_fresh_interpreter():
    code = ("import gin, github.run.run_test\n"
            "gin.parse_config_files_and_bindings(['./config/test.gin'], None)\n"
            "q = lambda k: gin.query_parameter(str(gin.query_parameter(k)))\n"
            "print(q('Controller.pv_num'), q('run_test.num_episode'))\n")
    r = _run_in_github(code)
    assert r.returncode == 0, r.stderr[-800:]
    assert r.stdout.strip().splitlines()[-1] == "500000 2"


def test_test_gin_parses_in_process_and_reaches_objects():
    from github.simul_bidding_env.Environment.BiddingEnv import BiddingEnv

    gin.parse_config_files_and_bindings([str(GIN_FILE)], None)
    env = BiddingEnv()
    assert (env.reserve_pv_price, env.min_remaining_budget) == (0.0001, 0.1)
    assert q(f"{CTRL}.pv_num") == 500000
    assert q(f"{CTRL}.num_agent_category") == 8
    assert q(f"{CTRL}.num_category") == 6
    assert q(f"{RUN}.num_tick") == 48
    assert q(f"{RUN}.generate_log") is False


def test_helpers_bindings_equal_test_gin():
    """テストで使う helpers.gin_bindings() の既定値が、本家 test.gin と同じ設定であること。"""
    keys = [f"{RUN}.generate_log", f"{RUN}.num_episode", f"{RUN}.num_tick", f"{ENV}.reserve_pv_price",
            f"{ENV}.min_remaining_budget", f"{CTRL}.num_agent_category", f"{CTRL}.num_category",
            f"{CTRL}.num_tick", f"{CTRL}.pv_num", f"{CTRL}.pv_generator_type"]
    gin.parse_config_files_and_bindings([str(GIN_FILE)], None)
    a = [q(k) for k in keys]
    gin.clear_config()
    apply_gin()
    assert a == [q(k) for k in keys]


# ---------- 完全修飾名の gin 束縛 ----------
def test_qualified_bindings_parse():
    gin.parse_config(gin_bindings())


@pytest.mark.parametrize("kw,attr,expected", [
    (dict(reserve_pv_price=0.0001), "reserve_pv_price", 0.0001),
    (dict(reserve_pv_price=0.05), "reserve_pv_price", 0.05),
    (dict(min_remaining_budget=0.1), "min_remaining_budget", 0.1),
    (dict(min_remaining_budget=1.0), "min_remaining_budget", 1.0),
])
def test_qualified_bindings_reach_bidding_env(kw, attr, expected):
    from github.simul_bidding_env.Environment.BiddingEnv import BiddingEnv

    apply_gin(**kw)
    assert getattr(BiddingEnv(), attr) == expected


def test_bidding_env_defaults_without_gin_differ_from_test_gin():
    """コード側の既定値(reserve=0.01)と test.gin の値(0.0001)は違う。gin を通さないと別の環境になる。"""
    from github.simul_bidding_env.Environment.BiddingEnv import BiddingEnv

    env = BiddingEnv()
    assert env.reserve_pv_price == 0.01 and env.min_remaining_budget == 0.1


def test_qualified_bindings_reach_controller():
    from github.simul_bidding_env.Controller.Controller import Controller
    from github.simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy

    apply_gin(pv_num=1500)
    c = Controller(player_agent=PidBiddingStrategy())
    assert (c.num_agent_category, c.num_category, c.num_tick, c.pv_num) == (8, 6, 48, 1500)
    assert c.num_agent == 48 and c.pv_generator_type == "neuripsPvGen"


def test_controller_defaults_without_gin_are_the_competition_setting():
    """gin を通さなくても、Controller の既定値は本番設定（48 エージェント・48 tick）で BiddingEnv(48 固定)と揃う。"""
    from github.simul_bidding_env.Controller.Controller import Controller
    from github.simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy

    c = Controller(player_agent=PidBiddingStrategy(), pv_num=1000)
    assert (c.num_agent, c.num_tick) == (48, 48)


def test_qualified_bindings_reach_run_test():
    apply_gin(num_episode=5, num_tick=12, generate_log=True)
    assert gin.query_parameter(f"{RUN}.num_episode") == 5
    assert gin.query_parameter(f"{RUN}.num_tick") == 12
    assert gin.query_parameter(f"{RUN}.generate_log") is True


def test_clear_config_resets_bindings():
    apply_gin(num_episode=7)
    gin.clear_config()
    with pytest.raises(ValueError):
        gin.query_parameter(f"{RUN}.num_episode")


def test_unknown_parameter_binding_is_rejected():
    with pytest.raises(ValueError):
        gin.parse_config(f"{CTRL}.no_such_param = 1")


def test_unqualified_selector_is_unambiguous_before_second_registration():
    """プロセス内で github.* だけを import した状態なら、短い名前でも解決できる（二重登録が起きた時だけ ambiguous）。"""
    gin.parse_config("run_test.num_episode = 3")
    assert gin.query_parameter(f"{RUN}.num_episode") == 3
