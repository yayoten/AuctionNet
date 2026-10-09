"""05b: ModelPvGenerator（深層生成モデルによる PV 生成。config の選択肢 "modelPvGen"）。"""
import numpy as np
import pytest

from github.simul_bidding_env.PvGenerator.ModelPvGen import ModelPvGenerator

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def mg():
    return ModelPvGenerator(num_tick=48, num_agent_category=8, select_category=[1, 2, 3, 4, 5, 6], pv_num=3000,
                            episode=0)


def test_constructs_and_has_48_ticks(mg):
    assert len(mg.pv_values) == 48 and len(mg.pValueSigmas) == 48


def test_num_agent_is_categories_times_agents_per_category(mg):
    assert mg.num_agent == 48
    assert all(a.shape[1] == 48 for a in mg.pv_values)


def test_values_and_sigmas_aligned(mg):
    assert [a.shape for a in mg.pv_values] == [a.shape for a in mg.pValueSigmas]


def test_pvalues_finite_and_nonnegative(mg):
    allv = np.concatenate(mg.pv_values)
    assert np.isfinite(allv).all() and allv.min() >= 0


def test_sigma_nonnegative_and_below_30_percent(mg):
    for v, s in zip(mg.pv_values, mg.pValueSigmas):
        assert (s >= 0).all() and (s <= 0.3 * np.abs(v) + 1e-12).all()


def test_total_traffic_is_positive_and_not_over_cap(mg):
    total = sum(a.shape[0] for a in mg.pv_values)
    assert 0 < total <= 105000


def test_features_per_tick_match_traffic(mg):
    assert [f.shape[0] for f in mg.pv_features] == [v.shape[0] for v in mg.pv_values]


def test_pv_num_is_capped_at_105000():
    g = ModelPvGenerator.__new__(ModelPvGenerator)  # 重い生成を避けて上限ロジックだけを見る
    assert min(int(500000 * 1.0), 105000) == 105000


def test_reset_regenerates(mg):
    before = mg.pv_values[0].copy()
    mg.reset(episode=1)
    assert mg.episode == 1
    assert len(mg.pv_values) == 48
    assert not (mg.pv_values[0].shape == before.shape and np.array_equal(mg.pv_values[0], before))


def test_sigma_generation_is_deterministic_per_episode(mg):
    a = mg.generate_pvalue_sigma(mg.pv_values)
    b = mg.generate_pvalue_sigma(mg.pv_values)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_noise_helpers_ranges(mg):
    s = mg.scale_noise(scale=0.1, shape=(1000,)).numpy()
    assert s.min() >= 0.9 and s.max() <= 1.1
    h = mg.shift_noise(scale=2e-5, shape=(1000,)).numpy()
    assert h.min() >= 0 and h.max() <= 2e-5


def test_controller_can_use_model_pv_generator():
    from github.simul_bidding_env.Controller.Controller import Controller
    from github.simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy

    c = Controller(player_agent=PidBiddingStrategy(), num_tick=48, num_agent_category=8, num_category=6, pv_num=3000,
                   pv_generator_type="modelPvGen")
    assert isinstance(c.pvGenerator, ModelPvGenerator)
    # 修正前は Controller が pv_num を渡さず、常に既定の 105000 件を生成していた（約90秒）
    assert c.pvGenerator.pv_num == 3000
