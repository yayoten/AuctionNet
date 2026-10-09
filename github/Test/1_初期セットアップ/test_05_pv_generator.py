"""05: NeurIPSPvGen（PV＝広告機会の生成）。"""
import numpy as np
import pytest

from github.simul_bidding_env.PvGenerator.NeurIPSPvGen import NeurIPSPvGen

PV = 2000


def gen(episode=0, pv_num=PV, **kw):
    return NeurIPSPvGen(episode=episode, num_tick=48, num_agent=48, num_agent_category=8, num_category=6,
                        pv_num=pv_num, **kw)


@pytest.fixture(scope="module")
def g0():
    return gen(0)


def test_returns_one_array_per_tick(g0):
    assert len(g0.pv_values) == 48 and len(g0.pValueSigmas) == 48


def test_every_tick_array_has_one_column_per_agent(g0):
    assert all(a.shape[1] == 48 for a in g0.pv_values)
    assert all(a.shape[1] == 48 for a in g0.pValueSigmas)


def test_sigma_shape_matches_value_shape(g0):
    assert [a.shape for a in g0.pv_values] == [a.shape for a in g0.pValueSigmas]


def test_total_traffic_is_close_to_pv_num(g0):
    total = sum(a.shape[0] for a in g0.pv_values)
    assert PV - 48 <= total <= PV


def test_traffic_ratio_base_sums_to_one(g0):
    assert g0.traffic_num_ratio_base.sum() == pytest.approx(1.0, abs=1e-9)
    assert len(g0.traffic_num_ratio_base) == 48


def test_perturbed_traffic_ratio_still_sums_to_one(g0):
    r = g0.generate_perturb_with_normalize(g0.traffic_num_ratio_base, 0.4, 0.4, 4, 0)
    assert r.sum() == pytest.approx(1.0, abs=1e-9)


def test_pvalues_are_probabilities(g0):
    allv = np.concatenate(g0.pv_values)
    assert allv.min() >= 0.0 and allv.max() <= 1.0 and np.isfinite(allv).all()


def test_pvalue_mean_is_near_base_level(g0):
    """base=0.0005。カテゴリ・tick の摂動で多少ずれるので、緩い範囲で確認。"""
    m = np.concatenate(g0.pv_values).mean()
    assert 0.0001 < m < 0.002


def test_sigma_is_nonnegative_and_bounded_by_30_percent_of_pvalue(g0):
    for v, s in zip(g0.pv_values, g0.pValueSigmas):
        assert (s >= 0).all()
        assert (s <= 0.3 * v + 1e-12).all()


def test_sigma_is_finite(g0):
    assert all(np.isfinite(s).all() for s in g0.pValueSigmas)


def test_same_episode_is_deterministic():
    a, b = gen(3), gen(3)
    assert all(np.array_equal(x, y) for x, y in zip(a.pv_values, b.pv_values))
    assert all(np.array_equal(x, y) for x, y in zip(a.pValueSigmas, b.pValueSigmas))


def test_different_episodes_differ():
    a, b = gen(0), gen(1)
    assert not np.array_equal(a.pv_values[10], b.pv_values[10])


def test_different_episodes_have_different_tick_traffic_profile():
    a = [x.shape[0] for x in gen(0).pv_values]
    b = [x.shape[0] for x in gen(1).pv_values]
    assert a != b


@pytest.mark.parametrize("pv_num", [500, 1000, 5000])
def test_pv_num_scales_traffic(pv_num):
    g = gen(0, pv_num=pv_num)
    total = sum(a.shape[0] for a in g.pv_values)
    assert pv_num - 48 <= total <= pv_num


def test_traffic_profile_follows_base_shape():
    """tick 0 は終盤より多い／深夜帯(tick 5-8)は少ない、という元の時間帯分布が残っている。"""
    g = gen(0, pv_num=20000)
    n = np.array([a.shape[0] for a in g.pv_values], dtype=float)
    assert n[:1].mean() > n[5:8].mean()
    assert n[16:24].mean() > n[5:8].mean()


def test_agent_zero_in_each_category_uses_category_base_mean():
    """各カテゴリの先頭エージェント(i % 8 == 0)は摂動なし = カテゴリ平均そのもの。"""
    g = gen(0)
    mean_all = g.generate_pvalue_mean()
    cat = g.calculate_pvalue_mean_by_category()
    for c in range(6):
        assert np.allclose(mean_all[c * 8], cat[c] * g.pvalue_mean_base)


def test_pvalue_mean_matrix_shape_and_positive():
    g = gen(0)
    m = g.generate_pvalue_mean()
    assert m.shape == (48, 48) and (m > 0).all() and (m <= 1).all()


def test_pvalue_std_ratio_in_range():
    r = gen(0).generate_pvalue_std_ratio()
    assert r.shape == (48, 48) and (r >= 0).all() and (r <= 1).all()


def test_generate_perturb_no_normalize_is_blockwise_constant_factor():
    g = gen(0)
    base = np.ones(48)
    out = g.generate_perturb_no_normalize(base, 0.7, 0.7, 8, 5)
    for k in range(6):
        block = out[k * 8:(k + 1) * 8]
        assert np.allclose(block, block[0])
    assert (out >= 0.3 - 1e-12).all() and (out <= 1.7 + 1e-12).all()


def test_generate_perturb_with_normalize_blockwise_factor_and_total():
    g = gen(0)
    out = g.generate_perturb_with_normalize(g.traffic_num_ratio_base, 0.4, 0.4, 4, 2)
    ratio = out / g.traffic_num_ratio_base
    for k in range(12):
        assert np.allclose(ratio[k * 4:(k + 1) * 4], ratio[k * 4])


@pytest.mark.parametrize("num_tick,num_agent,nac,nc", [(24, 30, 6, 5), (48, 48, 8, 6), (48, 24, 8, 3)])
def test_alternative_dimensions_construct(num_tick, num_agent, nac, nc):
    g = NeurIPSPvGen(episode=0, num_tick=num_tick, num_agent=num_agent, num_agent_category=nac, num_category=nc,
                     pv_num=800)
    assert len(g.pv_values) == num_tick
    assert all(a.shape[1] == num_agent for a in g.pv_values)


def test_default_constructor_args_are_the_competition_setting():
    import inspect

    sig = inspect.signature(NeurIPSPvGen.__init__).parameters
    assert (sig["num_tick"].default, sig["num_agent"].default, sig["num_agent_category"].default,
            sig["num_category"].default, sig["pv_num"].default) == (48, 48, 8, 6, 500000)


def test_reset_matches_fresh_instance_of_that_episode():
    g = gen(0)
    g.reset(episode=2)
    f = gen(2)
    assert all(np.array_equal(x, y) for x, y in zip(g.pv_values, f.pv_values))


# ---------- 修正済みの不具合（回帰防止） ----------
@pytest.mark.parametrize("pv_num", [500, 1000, 5000])
def test_reset_preserves_pv_num(pv_num):
    """修正前は reset() が __init__(episode=...) だけを呼び、pv_num が 500000 に戻っていた。"""
    g = gen(0, pv_num=pv_num)
    g.reset(episode=1)
    assert g.PV_NUM == pv_num
    assert pv_num - 48 <= sum(a.shape[0] for a in g.pv_values) <= pv_num


def test_reset_preserves_all_dimensions():
    g = NeurIPSPvGen(episode=0, num_tick=24, num_agent=30, num_agent_category=6, num_category=5, pv_num=800)
    g.reset(episode=3)
    assert (g.NUM_TICK, g.NUM_AGENT, g.NUM_AGENT_CATEGORY, g.NUM_CATEGORY, g.PV_NUM, g.episode) == (24, 30, 6, 5, 800, 3)
    assert len(g.pv_values) == 24 and g.pv_values[0].shape[1] == 30


def test_reset_default_episode_is_zero():
    g = gen(4)
    g.reset()
    assert g.episode == 0


@pytest.mark.slow
def test_module_level_test_function_runs():
    """モジュール末尾の test()（7 エピソード×500000 PV を生成して表示）が動く。"""
    from github.simul_bidding_env.PvGenerator import NeurIPSPvGen as mod

    mod.test()


def test_class_has_pValueSigmas_not_pvalue_sigmas(g0):
    assert hasattr(g0, "pValueSigmas") and not hasattr(g0, "pvalue_sigmas")
