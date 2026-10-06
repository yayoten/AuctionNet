"""17: 直書きだった値を引数（既定値は従来どおり）にした変更の検証。

- 既定値のままなら、従来と挙動が変わらない（回帰）。
- 引数で変えると、挙動が実際に変わる（値が届いている）。
- run_test の player_agent / episode_ids / tick_hook が、結果を変えずに使える。
"""
import numpy as np
import pytest

import github.run.run_test as rt
from github.simul_bidding_env.Controller.Controller import Controller
from github.simul_bidding_env.Environment.BiddingEnv import BiddingEnv
from github.simul_bidding_env.Tracker.PlayerAnalysis import PlayerAnalysis
from github.simul_bidding_env.strategy.abid_bidding_strategy import AbidBiddingStrategy
from github.simul_bidding_env.strategy.onlinelp_bidding_strategy import OnlineLpBiddingStrategy
from github.simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
from helpers import make_pvalues

PV = 2000


def _pid_alphas(**kw):
    """予算の消費を人工的に変えながら bidding を呼び、alpha の系列を返す。"""
    s = PidBiddingStrategy(budget=100, **kw)
    s.reset()
    p, sg = make_pvalues(10)
    alphas = []
    for t, spent in enumerate([0, 1, 1, 30, 30, 1]):  # 少なすぎる→そのまま→多すぎる→…の順に刺激する
        s.remaining_budget -= spent
        s.bidding(t, p, sg, [], [], [], [], [])
        alphas.append(s.alpha)
    return alphas


# ---------- PID ----------
def test_pid_defaults_are_the_old_hardcoded_values():
    s = PidBiddingStrategy()
    assert (s.base_action, s.up_factor, s.down_factor, s.low_threshold, s.high_threshold) == (15, 1.2, 0.7, 0.7, 1.1)
    assert _pid_alphas()[0] == 15


def test_pid_base_action_reaches_the_first_alpha():
    assert _pid_alphas(base_action=40)[0] == 40


def test_pid_factors_and_thresholds_change_the_alpha_path():
    base = _pid_alphas()
    assert _pid_alphas(up_factor=1.5) != base
    assert _pid_alphas(down_factor=0.5) != base
    assert _pid_alphas(low_threshold=0.0, high_threshold=1e9)[1:] == [15] * 5  # どちらの条件にも入らない → alpha 不変


# ---------- ABid ----------
def test_abid_bid_scale_multiplies_bids_and_defaults_to_one():
    p, sg = make_pvalues(10)
    a, b = AbidBiddingStrategy(), AbidBiddingStrategy(bid_scale=2.5)
    assert a.bid_scale == 1.0
    ba, bb = a.bidding(0, p, sg, [], [], [], [], []), b.bidding(0, p, sg, [], [], [], [], [])
    np.testing.assert_allclose(bb, 2.5 * ba)


# ---------- OnlineLP ----------
def test_onlinelp_cap_ratio_bounds_the_alpha():
    p, sg = make_pvalues(10)
    for ratio in (1.5, 0.5):
        s = OnlineLpBiddingStrategy(cpa=2, cpa_cap_ratio=ratio)
        s.reset()
        bids = s.bidding(0, p, sg, [], [], [], [], [])
        assert (bids <= 2 * ratio * p + 1e-12).all()
    assert OnlineLpBiddingStrategy().cpa_cap_ratio == 1.5


# ---------- 環境・スコア・コントローラ ----------
def test_bidding_env_new_args_defaults_and_override():
    e = BiddingEnv()
    assert list(e.slot_coefficients) == [1, 0.8, 0.6] and e.DEFAULT_SEED == 1 and e.CONVERSION_SEED == 2
    e = BiddingEnv(slot_coefficients=(1, 0.5, 0.2), default_seed=7, conversion_seed=9)
    assert list(e.slot_coefficients) == [1, 0.5, 0.2] and e.DEFAULT_SEED == 7 and e.CONVERSION_SEED == 9


def test_player_analysis_beta_and_normalizer():
    a = PlayerAnalysis("x")
    assert a._get_score_neurips(10, 2.0, 1.0) == pytest.approx(10 * 0.25)  # beta=2 → (1/2)^2
    b = PlayerAnalysis("x", penalty_beta=1)
    assert b._get_score_neurips(10, 2.0, 1.0) == pytest.approx(10 * 0.5)
    assert PlayerAnalysis("x").score_normalizer == 20000 and PlayerAnalysis("x", score_normalizer=1).score_normalizer == 1


def test_controller_budget_ratio_scales_budgets(bind_small_gin):
    bind_small_gin(pv_num=PV)
    base = np.array(Controller(player_agent=PidBiddingStrategy()).budget_list, dtype=float)
    scaled = np.array(Controller(player_agent=PidBiddingStrategy(), budget_ratio=0.5).budget_list, dtype=float)
    np.testing.assert_allclose(scaled, base * 0.5)
    assert base[0] == 2900


# ---------- run_test の追加引数 ----------
def _run(bind, **kw):
    bind(pv_num=PV, num_episode=2)
    return rt.run_test(player_index=0, **kw)


def test_run_test_with_hook_gives_identical_result_and_calls_each_tick(bind_small_gin):
    ref = _run(bind_small_gin)
    calls = []
    got = _run(bind_small_gin, tick_hook=calls.append)
    assert len(calls) == 2 * 48
    assert (calls[0]["episode"], calls[0]["tick"], calls[-1]["episode"], calls[-1]["tick"]) == (0, 0, 1, 47)
    assert got["score"] == ref["score"] and got["reward"] == ref["reward"]
    assert {"bids", "pv_values", "slot", "cost", "reward", "remaining_budget_before", "agents"} <= set(calls[0])


def test_run_test_with_explicit_agent_equals_default_pid_fallback(bind_small_gin):
    ref = _run(bind_small_gin)
    agent = PidBiddingStrategy(exp_tempral_ratio=np.ones(48))
    agent.name += "0"
    got = _run(bind_small_gin, player_agent=agent)
    assert got["score"] == ref["score"] and got["reward"] == ref["reward"]


def test_run_test_episode_ids_select_the_episodes(bind_small_gin):
    r = _run(bind_small_gin, episode_ids=[3, 5])
    assert [x["episode"] for x in r["rawData"]] == [3, 5]


def test_different_param_gives_different_result(bind_small_gin):
    ref = _run(bind_small_gin, player_agent=PidBiddingStrategy(base_action=15))
    other = _run(bind_small_gin, player_agent=PidBiddingStrategy(base_action=120))
    assert other["allCost"] != ref["allCost"]
