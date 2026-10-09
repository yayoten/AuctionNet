"""07: ルールベース戦略（PID / ABid / OnlineLP）と戦略の基底クラス。"""
import numpy as np
import pandas as pd
import pytest

from conftest import SIM_DIR
from helpers import make_history, make_pvalues
from github.simul_bidding_env.strategy.abid_bidding_strategy import AbidBiddingStrategy
from github.simul_bidding_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from github.simul_bidding_env.strategy.onlinelp_bidding_strategy import OnlineLpBiddingStrategy
from github.simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy


# ---------- 基底クラス ----------
def test_base_strategy_is_abstract():
    with pytest.raises(TypeError):
        BaseBiddingStrategy()


def test_base_strategy_subclass_needs_both_methods():
    class OnlyReset(BaseBiddingStrategy):
        def reset(self):
            pass

    with pytest.raises(TypeError):
        OnlyReset()


def test_base_strategy_init_fields():
    class S(BaseBiddingStrategy):
        def reset(self):
            self.remaining_budget = self.budget

        def bidding(self, *a):
            return a[1]

    s = S(budget=7, name="n", cpa=3, category=2)
    assert (s.budget, s.remaining_budget, s.name, s.cpa, s.category) == (7, 7, "n", 3, 2)


# ---------- PID ----------
def make_pid(budget=100.0, **kw):
    s = PidBiddingStrategy(budget=budget, **kw)
    return s


def test_pid_defaults():
    s = PidBiddingStrategy()
    assert (s.budget, s.remaining_budget, s.name, s.cpa, s.category) == (100, 100, "PidBiddingStrategy", 1, 0)
    assert s.base_action == 15 and s.alpha is None
    assert s.exp_budget_ratio.shape == (48,)


def test_pid_tick0_alpha_is_base_action():
    s = make_pid()
    p, sg = make_pvalues(10)
    bids = s.bidding(0, p, sg, [], [], [], [], [])
    assert s.alpha == 15
    assert np.allclose(bids, 15 * p)


def test_pid_bids_are_alpha_times_pvalue_nonnegative():
    s = make_pid()
    p, sg = make_pvalues(100)
    assert (s.bidding(0, p, sg, [], [], [], [], []) >= 0).all()


def test_pid_underspend_raises_alpha_by_1_2():
    s = make_pid()
    p, sg = make_pvalues(10)
    s.bidding(0, p, sg, [], [], [], [], [])
    # 何も使っていない → last_tick_cost = 0 → 0 < 0.7 → alpha *= 1.2
    s.bidding(1, p, sg, [], [], [], [], [])
    assert s.alpha == pytest.approx(18.0)


def test_pid_overspend_lowers_alpha_by_0_7():
    s = make_pid()
    p, sg = make_pvalues(10)
    s.bidding(0, p, sg, [], [], [], [], [])
    s.remaining_budget = 50.0  # 1 tick で半分使った
    s.bidding(1, p, sg, [], [], [], [], [])
    assert s.alpha == pytest.approx(15 * 0.7)


def test_pid_on_pace_keeps_alpha():
    s = make_pid()
    p, sg = make_pvalues(10)
    s.bidding(0, p, sg, [], [], [], [], [])
    # tick1: cost * (48-1)/remaining を 0.7〜1.1 の間にする → cost = 100/47 * 0.9 ≒ 1.915
    cost = 100 / 47 * 0.9
    s.remaining_budget = 100 - cost
    s.bidding(1, p, sg, [], [], [], [], [])
    assert s.alpha == pytest.approx(15.0)


def test_pid_alpha_compounds_over_ticks():
    s = make_pid()
    p, sg = make_pvalues(10)
    s.bidding(0, p, sg, [], [], [], [], [])
    for t in range(1, 4):
        s.bidding(t, p, sg, [], [], [], [], [])  # 予算を使わない
    assert s.alpha == pytest.approx(15 * 1.2 ** 3)


def test_pid_reset_restores_budget():
    s = make_pid()
    s.remaining_budget = 3
    s.reset()
    assert s.remaining_budget == 100


def test_pid_reset_resets_last_remaining_budget():
    s = make_pid()
    p, sg = make_pvalues(10)
    s.bidding(0, p, sg, [], [], [], [], [])
    s.remaining_budget = 40
    s.bidding(1, p, sg, [], [], [], [], [])
    s.reset()
    assert s.last_remaining_budget == s.budget


def test_pid_budget_assigned_after_construction_then_reset_tracks_spend_correctly():
    """Controller と同じく budget を後から代入して reset() する流れ。修正前は last_remaining_budget が 100 のままで、
    tick1 の消化額が負と誤判定されて 1.2 倍されていた。"""
    s = make_pid()
    s.budget = 2900.0
    s.reset()
    p, sg = make_pvalues(10)
    s.bidding(0, p, sg, [], [], [], [], [])
    s.remaining_budget = 2800.0  # 100 使った → 100*47/2800 = 1.68 > 1.1 → 使いすぎ
    s.bidding(1, p, sg, [], [], [], [], [])
    assert s.alpha == pytest.approx(15 * 0.7)


def test_pid_second_episode_starts_clean():
    s = make_pid()
    p, sg = make_pvalues(10)
    s.bidding(0, p, sg, [], [], [], [], [])
    s.remaining_budget = 10.0
    s.bidding(1, p, sg, [], [], [], [], [])
    s.reset()
    s.bidding(0, p, sg, [], [], [], [], [])
    assert s.alpha == 15
    s.bidding(1, p, sg, [], [], [], [], [])  # 何も使っていない → 1.2 倍
    assert s.alpha == pytest.approx(18.0)


def test_pid_default_exp_ratio_arg_is_shared_mutable_default():
    """exp_tempral_ratio=np.ones(48) は共有ミュータブル既定値。"""
    a, b = PidBiddingStrategy(), PidBiddingStrategy()
    assert a.exp_budget_ratio is b.exp_budget_ratio


def test_pid_weighted_ratio_changes_underspend_threshold():
    r = np.ones(48)
    r[1:] = 0.01  # 以降の tick はほとんど配分が無い想定
    s = PidBiddingStrategy(budget=100, exp_tempral_ratio=r)
    p, sg = make_pvalues(5)
    s.bidding(0, p, sg, [], [], [], [], [])
    s.remaining_budget = 99.0  # cost=1
    s.bidding(1, p, sg, [], [], [], [], [])
    # 1 * r[1:].sum()/r[0]/99 = 0.47/99 < 0.7 → 1.2 倍
    assert s.alpha == pytest.approx(18.0)


# ---------- ABid ----------
def test_abid_bid_formula():
    s = AbidBiddingStrategy(cpa=2.0)
    p, sg = make_pvalues(20)
    bids = s.bidding(3, p, sg, [], [], [], [], [])
    assert np.allclose(bids, (1.0 * 2.0 / p.mean()) * p * p)


def test_abid_uses_tick_specific_base_action():
    r = np.linspace(0.5, 1.5, 48)
    s = AbidBiddingStrategy(cpa=1.0, exp_tempral_ratio=r)
    p, sg = make_pvalues(20)
    assert np.allclose(s.bidding(10, p, sg, [], [], [], [], []), r[10] / p.mean() * p * p)


def test_abid_bids_nonnegative_and_finite():
    s = AbidBiddingStrategy()
    p, sg = make_pvalues(50)
    b = s.bidding(0, p, sg, [], [], [], [], [])
    assert (b >= 0).all() and np.isfinite(b).all()


def test_abid_is_not_used_by_controller_pool():
    from github.simul_bidding_env.Controller import Controller as C
    import inspect

    assert "AbidBiddingStrategy" not in inspect.getsource(C)


# ---------- OnlineLP ----------
def lp(episode=0, category=0, budget=3000.0, cpa=100.0):
    s = OnlineLpBiddingStrategy(episode=episode)
    s.budget, s.cpa, s.category = budget, cpa, category
    s.reset()
    return s


def expected_alpha(df, tick, category, remaining, cpa):
    sub = df[(df.timeStepIndex == tick) & (df.advertiserCategoryIndex == category)]
    alpha = cpa
    over = sub[sub.cum_cost > remaining]
    if len(sub) and not over.empty:
        alpha = over.iloc[0]["realCPA"]
    return min(cpa * 1.5, alpha)


@pytest.mark.parametrize("episode", range(7))
def test_onlinelp_loads_each_episode_file(episode):
    s = OnlineLpBiddingStrategy(episode=episode)
    assert len(s.model) > 0


def test_onlinelp_missing_episode_file_raises():
    with pytest.raises(FileNotFoundError):
        OnlineLpBiddingStrategy(episode=99)


@pytest.mark.parametrize("episode,category,tick,remaining", [
    (0, 0, 0, 3000.0), (0, 0, 10, 1500.0), (1, 2, 5, 100.0), (3, 5, 47, 10.0), (6, 3, 20, 5000.0),
    (2, 1, 30, 0.5), (4, 4, 0, 1e9),
])
def test_onlinelp_matches_independent_reimplementation(episode, category, tick, remaining):
    s = lp(episode, category)
    s.remaining_budget = remaining
    p, sg = make_pvalues(25)
    bids = s.bidding(tick, p, sg, [], [], [], [], [])
    df = pd.read_csv(SIM_DIR / "strategy/official_agent/onlineLpTest" / f"episode-{episode}.csv")
    assert np.allclose(bids, expected_alpha(df, tick, category, remaining, 100.0) * p)


def test_onlinelp_alpha_capped_at_1_5_cpa():
    s = lp(0, 0, cpa=1.0)
    s.remaining_budget = 0.0
    p, sg = make_pvalues(10)
    for t in range(48):
        assert (s.bidding(t, p, sg, [], [], [], [], []) <= 1.5 * p + 1e-12).all()


def test_onlinelp_falls_back_to_cpa_when_budget_exceeds_all_cum_costs():
    s = lp(0, 0, cpa=100.0)
    s.remaining_budget = 1e12
    p, sg = make_pvalues(10)
    assert np.allclose(s.bidding(0, p, sg, [], [], [], [], []), 100.0 * p)


def test_onlinelp_unknown_category_falls_back_to_cpa():
    s = lp(0, 0, cpa=80.0)
    s.category = 99
    p, sg = make_pvalues(10)
    assert np.allclose(s.bidding(0, p, sg, [], [], [], [], []), 80.0 * p)


def test_onlinelp_name_and_defaults():
    s = OnlineLpBiddingStrategy()
    assert s.name == "OnlineLpBiddingStrategy" and (s.budget, s.cpa) == (100, 2)


def test_onlinelp_reset_restores_budget():
    s = lp()
    s.remaining_budget = 1
    s.reset()
    assert s.remaining_budget == s.budget


# ---------- 共通: どのルールベースでも bidding の形式を守るか ----------
@pytest.mark.parametrize("make", [
    lambda: PidBiddingStrategy(), lambda: AbidBiddingStrategy(), lambda: lp(),
], ids=["pid", "abid", "onlinelp"])
@pytest.mark.parametrize("n", [1, 7, 500])
def test_bidding_returns_array_of_same_length(make, n):
    s = make()
    p, sg = make_pvalues(n)
    out = s.bidding(0, p, sg, [], [], [], [], [])
    assert np.asarray(out).shape == (n,)


@pytest.mark.parametrize("make", [lambda: PidBiddingStrategy(), lambda: lp()], ids=["pid", "onlinelp"])
def test_bidding_accepts_full_history_arguments(make):
    s = make()
    p, sg = make_pvalues(30)
    h = make_history(5, 30)
    s.bidding(0, p, sg, [], [], [], [], [])
    out = s.bidding(5, p, sg, *h)
    assert out.shape == (30,)
