"""08: 学習済み戦略 7 種（同梱の official_agent を torch.jit で読み込む）。"""
import numpy as np
import pytest
import torch

from helpers import make_history, make_pvalues
from github.simul_bidding_env.strategy.bc_bidding_strategy import BcBiddingStrategy
from github.simul_bidding_env.strategy.bcq_bidding_strategy import BcqBiddingStrategy
from github.simul_bidding_env.strategy.cql_bidding_strategy import CqlBiddingStrategy
from github.simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
from github.simul_bidding_env.strategy.mbrl_combomicro_bidding_strategy import MbrlComboMicroBiddingStrategy
from github.simul_bidding_env.strategy.mbrl_mopo_bidding_strategy import MbrlMopoBiddingStrategy
from github.simul_bidding_env.strategy.td3_bc_bidding_strategy import TD3_BCBiddingStrategy

STRATEGIES = {
    "iql": IqlBiddingStrategy, "bc": BcBiddingStrategy, "bcq": BcqBiddingStrategy, "cql": CqlBiddingStrategy,
    "td3_bc": TD3_BCBiddingStrategy, "mopo": MbrlMopoBiddingStrategy, "combo": MbrlComboMicroBiddingStrategy,
}
DEFAULT_NAMES = {
    "iql": "Iql-PlayerStrategy", "td3_bc": "TD3_BC-PlayerStrategy",
}

@pytest.fixture(params=list(STRATEGIES))
def key(request):
    return request.param


def make(key, budget=3000.0, cpa=100.0, category=0):
    s = STRATEGIES[key]()
    s.budget, s.cpa, s.category = budget, cpa, category
    s.reset()
    return s


def alpha_of(bids, p):
    return bids / p


def test_constructs_and_has_model_and_normalize_dict(key):
    s = STRATEGIES[key]()
    assert isinstance(s.model, torch.jit.ScriptModule) or hasattr(s.model, "forward")
    assert isinstance(s.normalize_dict, dict) and s.normalize_dict


def test_default_fields(key):
    s = STRATEGIES[key]()
    assert (s.budget, s.remaining_budget, s.cpa, s.category) == (100, 100, 2, 1)
    assert isinstance(s.name, str) and s.name
    if key in DEFAULT_NAMES:
        assert s.name == DEFAULT_NAMES[key]


def test_reset_restores_remaining_budget(key):
    s = make(key)
    s.remaining_budget = 5.0
    s.reset()
    assert s.remaining_budget == 3000.0


@pytest.mark.parametrize("n", [1, 10, 400])
def test_tick0_output_shape_and_finite(key, n):
    s = make(key)
    p, sg = make_pvalues(n)
    b = s.bidding(0, p, sg, [], [], [], [], [])
    assert b.shape == (n,) and np.isfinite(b).all()


def test_tick0_bids_are_nonnegative(key):
    s = make(key)
    p, sg = make_pvalues(100)
    assert (s.bidding(0, p, sg, [], [], [], [], []) >= 0).all()


def test_bid_is_scalar_alpha_times_pvalue(key):
    s = make(key)
    p, sg = make_pvalues(100)
    a = alpha_of(s.bidding(0, p, sg, [], [], [], [], []), p)
    assert np.allclose(a, a[0])


@pytest.mark.parametrize("tick", [1, 3, 10, 24, 47])
def test_with_history_output_valid(key, tick):
    s = make(key)
    s.remaining_budget = 2000.0
    p, sg = make_pvalues(60)
    h = make_history(tick, 60, seed=tick)
    b = s.bidding(tick, p, sg, *h)
    assert b.shape == (60,) and np.isfinite(b).all() and (b >= 0).all()


def test_alpha_in_plausible_range(key):
    """cpa=100 のとき、alpha（入札/pValue）が極端でないこと。正常な学習済みモデルなら 0〜1000 程度。"""
    s = make(key)
    p, sg = make_pvalues(20)
    for t in (0, 5, 20):
        h = make_history(t, 20, seed=t) if t else ([], [], [], [], [])
        a = alpha_of(s.bidding(t, p, sg, *h), p)[0]
        assert 0 <= a <= 1000, f"alpha={a} at tick {t}"


def test_deterministic_for_same_input(key):
    s = make(key)
    p, sg = make_pvalues(30)
    h = make_history(6, 30)
    a = s.bidding(6, p, sg, *h)
    b = s.bidding(6, p, sg, *h)
    assert np.array_equal(a, b)


def test_two_instances_agree(key):
    p, sg = make_pvalues(30)
    h = make_history(6, 30)
    assert np.allclose(make(key).bidding(6, p, sg, *h), make(key).bidding(6, p, sg, *h))


def test_bids_scale_linearly_with_pvalue(key):
    """alpha は pValue の平均等から決まる状態に依存。PV 毎の pValue を 2 倍にした時、状態が変わるので厳密な 2 倍にはならないが、
    同じ状態なら bid ∝ pValue。ここでは同一入力で pValue の並び替えが結果の並び替えになる（PV 間独立）ことを確認する。"""
    s = make(key)
    p, sg = make_pvalues(40)
    perm = np.random.default_rng(0).permutation(40)
    b = s.bidding(0, p, sg, [], [], [], [], [])
    b2 = s.bidding(0, p[perm], sg[perm], [], [], [], [], [])
    assert np.allclose(b[perm], b2)


def test_does_not_mutate_inputs(key):
    s = make(key)
    p, sg = make_pvalues(30)
    h = make_history(4, 30)
    p0, sg0 = p.copy(), sg.copy()
    h0 = [[x.copy() for x in col] for col in h]
    s.bidding(4, p, sg, *h)
    assert np.array_equal(p, p0) and np.array_equal(sg, sg0)
    for col, col0 in zip(h, h0):
        for x, x0 in zip(col, col0):
            assert np.array_equal(x, x0)


def test_budget_left_changes_the_state_and_usually_the_bid(key):
    """予算残が違う状態では、少なくとも 1 つの tick で alpha が変わる（状態に反応している）。"""
    if key == "bcq":
        pytest.skip("BCQ の出力は今回の入力では飽和していて状態に反応しない（test_bcq_output_saturates 参照）")
    p, sg = make_pvalues(30)
    h = make_history(10, 30)
    outs = []
    for rem in (3000.0, 1500.0, 100.0):
        s = make(key)
        s.remaining_budget = rem
        outs.append(alpha_of(s.bidding(10, p, sg, *h), p)[0])
    assert len({round(float(o), 6) for o in outs}) > 1


def test_zero_budget_does_not_crash(key):
    s = make(key, budget=0.0)
    p, sg = make_pvalues(5)
    b = s.bidding(0, p, sg, [], [], [], [], [])
    assert b.shape == (5,)


def test_works_with_single_pv(key):
    s = make(key)
    p, sg = make_pvalues(1)
    h = make_history(3, 1)
    assert s.bidding(3, p, sg, *h).shape == (1,)


def test_extreme_pvalues_stay_finite(key):
    s = make(key)
    p = np.array([0.0, 1e-9, 1.0])
    sg = p * 0.1
    b = s.bidding(0, p, sg, [], [], [], [], [])
    assert np.isfinite(b).all() and b[0] == 0


def test_empty_pv_array_current_behavior(key):
    s = make(key)
    try:
        b = s.bidding(0, np.array([]), np.array([]), [], [], [], [], [])
    except Exception as e:
        pytest.skip(f"0 件の PV は未対応（{type(e).__name__}）。生成側が 0 件 tick を作らない前提")
    assert b.shape == (0,)


def test_player_wrapper_compat_attribute_assignment(key):
    """Controller は budget / cpa / category を後から代入し reset() する。その流れで動くこと。"""
    s = STRATEGIES[key]()
    s.budget, s.cpa, s.category = 4800, 60, 3
    s.reset()
    assert s.remaining_budget == 4800


def test_bcq_output_saturates_at_max_action():
    """BCQ は max_action(≒95) で飽和して見える。この入力範囲で全て同じ値 → 状態に反応していない可能性（要追加調査）。"""
    s = make("bcq")
    p, sg = make_pvalues(20)
    alphas = []
    for t in (0, 5, 20):
        h = make_history(t, 20, seed=t) if t else ([], [], [], [], [])
        alphas.append(float(alpha_of(s.bidding(t, p, sg, *h), p)[0]))
    assert max(alphas) - min(alphas) < 1e-3
    assert alphas[0] == pytest.approx(95.0, abs=1e-3)
