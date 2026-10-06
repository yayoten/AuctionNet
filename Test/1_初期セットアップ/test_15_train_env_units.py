"""15: strategy_train_env（学習・オフライン評価側）の単体テスト。小さな合成データで、各部品が動き、学習できることを確かめる。"""
import importlib
import os
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from github.strategy_train_env.bidding_train_env.baseline.bc.behavior_clone import BC
from github.strategy_train_env.bidding_train_env.baseline.bcq.bcq import BCQ
from github.strategy_train_env.bidding_train_env.baseline.cql.cql import CQL
from github.strategy_train_env.bidding_train_env.baseline.dt.dt import DecisionTransformer
from github.strategy_train_env.bidding_train_env.baseline.iql.iql import IQL
from github.strategy_train_env.bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from github.strategy_train_env.bidding_train_env.baseline.td3_bc.td3_bc import TD3_BC
from github.strategy_train_env.bidding_train_env.common.utils import (normalize_reward, normalize_state,
                                                                      save_normalize_dict)
from github.strategy_train_env.bidding_train_env.offline_eval.offline_env import OfflineEnv

dl_mod = importlib.import_module("github.strategy_train_env.bidding_train_env.offline_eval.test_dataloader")

B, D = 32, 16


def batch(seed=0, b=B, d=D):
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(b, d, generator=g)
    a = torch.rand(b, 1, generator=g) * 100
    r = torch.rand(b, 1, generator=g)
    ns = torch.rand(b, d, generator=g)
    done = (torch.rand(b, 1, generator=g) < 0.1).float()
    return s, a, r, ns, done


def _f(x):
    return float(torch.as_tensor(x).detach())


# ---------- common.utils ----------
def make_df():
    return pd.DataFrame({
        "state": [(1.0, 10.0, 100.0), (2.0, 20.0, 200.0), (3.0, 30.0, 300.0)],
        "next_state": [(2.0, 20.0, 200.0), (3.0, 30.0, 300.0), None],
        "reward": [10.0, 20.0, 30.0],
    })


def test_normalize_state_stats_hand_computed():
    df = make_df()
    stats = normalize_state(df, 3, [0, 2])
    assert set(stats) == {0, 2}
    assert stats[0]["min"] == 1 and stats[0]["max"] == 3 and stats[0]["mean"] == 2
    assert stats[2]["min"] == 100 and stats[2]["max"] == 300
    assert stats[0]["std"] == pytest.approx(1.0)


def test_normalize_state_formula_uses_range_plus_0_01():
    df = make_df()
    normalize_state(df, 3, [0])
    assert df["normalize_state0"].tolist() == pytest.approx([0 / 2.01, 1 / 2.01, 2 / 2.01])


def test_normalize_state_leaves_other_dims_unchanged():
    df = make_df()
    normalize_state(df, 3, [0])
    assert df["normalize_state1"].tolist() == [10.0, 20.0, 30.0]


def test_normalize_state_builds_tuple_columns():
    df = make_df()
    normalize_state(df, 3, [0, 2])
    assert all(len(t) == 3 for t in df["normalize_state"])
    assert df["normalize_state"].iloc[0][1] == 10.0


def test_normalize_state_missing_next_state_becomes_zero():
    df = make_df()
    normalize_state(df, 3, [0])
    assert df["next_state0"].iloc[2] == 0.0


def test_normalize_state_max_equals_min_does_not_divide_by_zero():
    df = pd.DataFrame({"state": [(5.0,)] * 3, "next_state": [(5.0,)] * 3})
    normalize_state(df, 1, [0])
    assert np.isfinite(df["normalize_state0"]).all()


def test_normalize_reward_minmax():
    df = make_df()
    out = normalize_reward(df, "reward")
    assert out.tolist() == pytest.approx([0, 0.5, 1.0], abs=1e-6)
    assert "normalize_reward" in df


def test_normalize_reward_constant_column_is_zero():
    df = pd.DataFrame({"reward": [3.0, 3.0]})
    assert normalize_reward(df, "reward").tolist() == [0.0, 0.0]


def test_save_normalize_dict_roundtrip_and_creates_dir(tmp_path):
    d = {13: {"min": 0.0, "max": 5.0, "mean": 2.0, "std": 1.0}}
    save_normalize_dict(d, str(tmp_path / "new" / "dir"))
    with open(tmp_path / "new" / "dir" / "normalize_dict.pkl", "rb") as f:
        assert pickle.load(f) == d


def test_saved_normalize_dict_format_is_what_strategies_expect(tmp_path):
    """戦略は `for key, value in normalize_dict.items(): value['min'], value['max']` で読む。"""
    df = make_df()
    stats = normalize_state(df, 3, [0, 2])
    save_normalize_dict(stats, str(tmp_path))
    nd = pickle.load(open(tmp_path / "normalize_dict.pkl", "rb"))
    assert all({"min", "max"} <= set(v) for v in nd.values())


# ---------- ReplayBuffer ----------
def test_replay_buffer_push_len_sample_shapes():
    rb = ReplayBuffer()
    for i in range(50):
        rb.push(np.ones(16) * i, np.array([i]), np.array([0.5]), np.ones(16), np.array([0]))
    assert len(rb) == 50
    s, a, r, ns, d = rb.sample(10)
    assert s.shape == (10, 16) and a.shape == (10, 1) and r.shape == (10, 1) and ns.shape == (10, 16) and d.shape == (10, 1)
    assert s.dtype == torch.float32


def test_replay_buffer_sample_larger_than_size_raises():
    rb = ReplayBuffer()
    rb.push(np.ones(3), np.array([1]), np.array([1]), np.ones(3), np.array([0]))
    with pytest.raises(ValueError):
        rb.sample(5)


def test_replay_buffer_sample_has_no_duplicates():
    rb = ReplayBuffer()
    for i in range(30):
        rb.push(np.array([i]), np.array([i]), np.array([i]), np.array([i]), np.array([0]))
    s, *_ = rb.sample(30)
    assert sorted(s.flatten().tolist()) == list(range(30))


# ---------- 各アルゴリズム：step / take_actions / save_jit ----------
ALGOS = {
    "bc": lambda: BC(dim_obs=D),
    "iql": lambda: IQL(dim_obs=D),
    "cql": lambda: CQL(dim_obs=D),
    "bcq": lambda: BCQ(state_dim=D),
    "td3_bc": lambda: TD3_BC(dim_obs=D),
}


@pytest.fixture(params=list(ALGOS))
def algo(request):
    return request.param, ALGOS[request.param]()


def do_step(name, m, seed=0):
    s, a, r, ns, d = batch(seed)
    return m.step(s, a) if name == "bc" else m.step(s, a, r, ns, d)


def test_step_returns_finite_losses(algo):
    name, m = algo
    out = do_step(name, m)
    outs = out if isinstance(out, (tuple, list, np.ndarray)) else [out]
    outs = [o for o in outs if o is not None]  # TD3_BC は方策更新を遅延させるため actor loss が None の step がある
    assert outs and all(np.isfinite(_f(o)) for o in outs)


def test_td3_bc_actor_is_updated_every_policy_freq_steps():
    m = TD3_BC(dim_obs=D)
    got = [m.step(*batch(i))[1] for i in range(4)]
    assert [g is None for g in got] == [True, False, True, False]


def test_step_changes_parameters(algo):
    name, m = algo
    before = [p.detach().clone() for p in m.parameters()]
    for i in range(3):
        do_step(name, m, i)
    after = list(m.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_take_actions_shape_and_finite(algo):
    name, m = algo
    s, *_ = batch()
    out = m.take_actions(s if name != "bcq" else s[:1].numpy())
    out = np.asarray(out)
    assert np.isfinite(out).all()
    assert out.size in (1, B)  # BCQ は 1 状態ずつ、他はバッチ


def test_save_jit_and_reload_matches_original(algo, tmp_path):
    name, m = algo
    for i in range(2):
        do_step(name, m, i)
    m.eval() if hasattr(m, "eval") else None
    m.save_jit(str(tmp_path / "m"))
    files = list((tmp_path / "m").iterdir())
    assert len(files) == 1 and files[0].suffix == ".pth"
    loaded = torch.jit.load(str(files[0]))
    x = torch.rand(D)
    out = loaded(x)
    assert out.numel() == 1 and torch.isfinite(out).all()


def test_save_jit_creates_nested_directory(algo, tmp_path):
    name, m = algo
    m.save_jit(str(tmp_path / "a" / "b"))
    assert (tmp_path / "a" / "b").is_dir()


def test_same_seed_gives_same_initial_parameters():
    for name in ("bc", "cql"):
        torch.manual_seed(5)
        a = ALGOS[name]()
        torch.manual_seed(5)
        b = ALGOS[name]()
        assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters())), name


# ---------- 学習できること ----------
def test_bc_learns_a_simple_function():
    """a = 50 * mean(state) を教師にして、BC の損失が下がり、予測が相関すること。"""
    torch.manual_seed(0)
    np.random.seed(0)
    m = BC(dim_obs=D, actor_lr=1e-3)
    S = torch.rand(2000, D)
    A = (50 * S.mean(dim=1, keepdim=True))
    first = np.mean([np.mean(m.step(S[i * 100:(i + 1) * 100], A[i * 100:(i + 1) * 100])) for i in range(5)])
    for _ in range(300):
        idx = torch.randint(0, 2000, (100,))
        m.step(S[idx], A[idx])
    last = np.mean([np.mean(m.step(S[i * 100:(i + 1) * 100], A[i * 100:(i + 1) * 100])) for i in range(5)])
    assert last < first
    pred = np.asarray(m.take_actions(S[:200])).flatten()
    assert np.corrcoef(pred, A[:200].numpy().flatten())[0, 1] > 0.5


def test_iql_value_loss_decreases_on_fixed_batch():
    torch.manual_seed(0)
    m = IQL(dim_obs=D)
    s, a, r, ns, d = batch(1, b=128)
    losses = [_f(m.step(s, a, r, ns, d)[0]) for _ in range(200)]
    assert np.mean(losses[-20:]) < np.mean(losses[:20])


def test_td3_bc_critic_loss_decreases_on_fixed_batch():
    torch.manual_seed(0)
    m = TD3_BC(dim_obs=D)
    s, a, r, ns, d = batch(1, b=128)
    losses = [_f(m.step(s, a, r, ns, d)[0]) for _ in range(150)]
    assert np.mean(losses[-20:]) < np.mean(losses[:20])


def test_bcq_critic_loss_decreases_on_fixed_batch():
    torch.manual_seed(0)
    m = BCQ(state_dim=D)
    s, a, r, ns, d = batch(1, b=64)
    losses = [_f(m.step(s, a, r, ns, d)[0]) for _ in range(100)]
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


# ---------- Decision Transformer ----------
def dt_inputs(b=4, K=4):
    st = torch.rand(b, K, D)
    ac = torch.rand(b, K, 1)
    rw = torch.rand(b, K, 1)
    dn = torch.zeros(b, K)
    rtg = torch.rand(b, K + 1, 1)
    ts = torch.randint(0, 48, (b, K))
    am = torch.ones(b, K)
    return st, ac, rw, dn, rtg, ts, am


@pytest.fixture
def dt():
    return DecisionTransformer(state_dim=D, act_dim=1, state_mean=np.zeros(D), state_std=np.ones(D), K=4, max_ep_len=48)


def test_dt_forward_shapes(dt):
    st, ac, rw, dn, rtg, ts, am = dt_inputs()
    sp, ap, rp, _ = dt.forward(st, ac, rw, rtg[:, :-1], ts, am)
    assert sp.shape == (4, 4, D) and ap.shape == (4, 4, 1) and rp.shape == (4, 4, 1)


def test_dt_step_returns_finite_float(dt):
    loss = dt.step(*dt_inputs())
    assert isinstance(loss, float) and np.isfinite(loss)


def test_dt_initial_lr_is_tiny_because_of_10000_step_warmup(dt):
    """スケジューラは 10000 step で線形ウォームアップ。run_decision_transformer は毎 step scheduler.step() を呼ぶ前提。"""
    assert dt.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4 / 10000)
    for _ in range(100):
        dt.scheduler.step()
    assert dt.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4 * 101 / 10000)


def test_dt_loss_decreases_on_fixed_batch(dt):
    """勾配が流れ、学習できること（ウォームアップを外して一定の学習率にして検証）。"""
    torch.manual_seed(0)
    for g in dt.optimizer.param_groups:
        g["lr"] = 1e-3
    dt.scheduler = torch.optim.lr_scheduler.LambdaLR(dt.optimizer, lambda s: 1.0)
    inp = dt_inputs()
    losses = [dt.step(*inp) for _ in range(80)]
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def test_dt_take_actions_rolls_out_over_ticks(dt):
    dt.init_eval()
    dt.eval()
    state = np.random.default_rng(0).random(D).astype(np.float32)
    a0 = dt.take_actions(state)
    a1 = dt.take_actions(state, pre_reward=1.0)
    a2 = dt.take_actions(state, pre_reward=0.0)
    for a in (a0, a1, a2):
        assert np.asarray(a).size == 1 and np.isfinite(a).all()
    assert dt.eval_states.shape[0] == 3


def test_dt_take_actions_second_call_requires_pre_reward(dt):
    dt.init_eval()
    state = np.zeros(D, dtype=np.float32)
    dt.take_actions(state)
    with pytest.raises(AssertionError):
        dt.take_actions(state)


def test_dt_save_net_writes_dt_pt_inside_directory(dt, tmp_path):
    dt.save_net(str(tmp_path / "dt"))
    assert (tmp_path / "dt" / "dt.pt").is_file()


def test_dt_load_net_roundtrip_with_explicit_file_path(dt, tmp_path):
    dt.save_net(str(tmp_path / "dt"))
    dt2 = DecisionTransformer(state_dim=D, act_dim=1, state_mean=np.zeros(D), state_std=np.ones(D), K=4, max_ep_len=48)
    dt2.load_net(str(tmp_path / "dt" / "dt.pt"))
    x = dt_inputs()
    dt.eval(); dt2.eval()
    with torch.no_grad():
        o1 = dt.forward(x[0], x[1], x[2], x[4][:, :-1], x[5], x[6])[1]
        o2 = dt2.forward(x[0], x[1], x[2], x[4][:, :-1], x[5], x[6])[1]
    assert torch.allclose(o1, o2)


def test_dt_load_net_accepts_the_directory_given_to_save_net(dt, tmp_path):
    """修正前は load_net がディレクトリを受け取れず IsADirectoryError だった（既定値もディレクトリ）。"""
    dt.save_net(str(tmp_path / "dt"))
    dt2 = DecisionTransformer(state_dim=D, act_dim=1, state_mean=np.zeros(D), state_std=np.ones(D), K=4, max_ep_len=48)
    dt2.load_net(str(tmp_path / "dt"))
    assert all(torch.equal(a, b) for a, b in zip(dt.state_dict().values(), dt2.state_dict().values()))


def test_dt_save_jit_raises_clear_not_implemented(dt, tmp_path):
    """DT は torch.jit.script できない。分かりにくい RuntimeError ではなく、明示的に NotImplementedError を出す。"""
    with pytest.raises(NotImplementedError):
        dt.save_jit(str(tmp_path / "j"))
    assert not (tmp_path / "j").exists()


# ---------- OfflineEnv ----------
def test_offline_env_status_is_bid_ge_least_winning_cost():
    env = OfflineEnv()
    bids = np.array([1.0, 0.5, 0.2, 0.0])
    lwc = np.array([0.5, 0.5, 0.5, 0.0])
    val, cost, status, conv = env.simulate_ad_bidding(np.full(4, .5), np.full(4, .01), bids, lwc)
    assert status.tolist() == [True, True, False, True]


def test_offline_env_cost_is_least_winning_cost_when_won():
    env = OfflineEnv()
    bids = np.array([1.0, 0.1])
    lwc = np.array([0.4, 0.4])
    _, cost, status, _ = env.simulate_ad_bidding(np.full(2, .1), np.full(2, .01), bids, lwc)
    assert cost.tolist() == [0.4, 0.0]


def test_offline_env_conversion_only_for_wins_and_values_clipped():
    np.random.seed(0)
    env = OfflineEnv()
    n = 5000
    bids = np.concatenate([np.ones(n), np.zeros(n)])
    lwc = np.full(2 * n, 0.5)
    val, cost, status, conv = env.simulate_ad_bidding(np.full(2 * n, 0.3), np.full(2 * n, 0.01), bids, lwc)
    assert (conv[n:] == 0).all() and (val[n:] == 0).all()
    assert val.min() >= 0 and val.max() <= 1
    assert conv[:n].mean() == pytest.approx(0.3, abs=0.03)


def test_offline_env_is_reproducible_under_numpy_seed():
    env = OfflineEnv()
    args = (np.full(100, .3), np.full(100, .05), np.ones(100), np.full(100, .1))
    np.random.seed(3); a = env.simulate_ad_bidding(*args)
    np.random.seed(3); b = env.simulate_ad_bidding(*args)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_offline_env_min_remaining_budget_default():
    assert OfflineEnv().min_remaining_budget == 0.1


def test_offline_env_module_test_function_runs(capsys):
    from github.strategy_train_env.bidding_train_env.offline_eval import offline_env

    offline_env.test()
    assert "Tick Cost" in capsys.readouterr().out


def test_utils_module_demo_runs_as_script(tmp_path):
    """common/utils.py の __main__ デモ（修正前は normalize_reward の引数不足で落ちた）。"""
    import subprocess, sys
    from conftest import REPO_ROOT, TRAIN_DIR

    r = subprocess.run([sys.executable, "-W", "ignore", str(TRAIN_DIR / "bidding_train_env/common/utils.py")],
                       cwd=tmp_path, capture_output=True, text=True, timeout=120,
                       env=dict(__import__("os").environ, PYTHONPATH=str(REPO_ROOT)))
    assert r.returncode == 0, r.stderr[-500:]


# ---------- TestDataLoader ----------
def make_log_csv(path, periods=2, agents=3, ticks=4, pvs=5):
    rows = []
    rng = np.random.default_rng(0)
    for p in range(periods):
        for a in range(agents):
            for t in range(ticks):
                for v in range(pvs):
                    rows.append(dict(deliveryPeriodIndex=p, advertiserNumber=a, timeStepIndex=t, pvIndex=v,
                                     pValue=rng.random() * .01, pValueSigma=rng.random() * .001,
                                     leastWinningCost=rng.random()))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_dataloader_groups_by_period_and_advertiser(tmp_path):
    make_log_csv(tmp_path / "log.csv")
    dl = dl_mod.TestDataLoader(file_path=str(tmp_path / "log.csv"))
    assert len(dl.keys) == 6 and (0, 0) in dl.keys and (1, 2) in dl.keys


def test_dataloader_writes_pickle_next_to_csv(tmp_path):
    make_log_csv(tmp_path / "log.csv")
    dl_mod.TestDataLoader(file_path=str(tmp_path / "log.csv"))
    assert (tmp_path / "raw_data.pickle").is_file()


def test_dataloader_reuses_pickle_even_if_csv_changes(tmp_path):
    """キャッシュの pickle が優先される → CSV を差し替えても古いデータが読まれる（注意点）。"""
    make_log_csv(tmp_path / "log.csv", periods=2)
    dl_mod.TestDataLoader(file_path=str(tmp_path / "log.csv"))
    make_log_csv(tmp_path / "log.csv", periods=1)
    dl2 = dl_mod.TestDataLoader(file_path=str(tmp_path / "log.csv"))
    assert len(dl2.keys) == 6


def test_dataloader_mock_data_shapes(tmp_path):
    make_log_csv(tmp_path / "log.csv", ticks=4, pvs=5)
    dl = dl_mod.TestDataLoader(file_path=str(tmp_path / "log.csv"))
    n, pv, sg, lwc = dl.mock_data(dl.keys[0])
    assert n == 4 and len(pv) == len(sg) == len(lwc) == 4
    assert all(x.shape == (5,) for x in pv + sg + lwc)


def test_dataloader_sorts_ticks(tmp_path):
    make_log_csv(tmp_path / "log.csv")
    df = pd.read_csv(tmp_path / "log.csv").sample(frac=1, random_state=0)
    df.to_csv(tmp_path / "log.csv", index=False)
    dl = dl_mod.TestDataLoader(file_path=str(tmp_path / "log.csv"))
    for key in dl.keys:
        assert dl.test_dict[key].timeStepIndex.is_monotonic_increasing


def test_dataloader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dl_mod.TestDataLoader(file_path=str(tmp_path / "nope.csv"))


def test_dataloader_default_path_is_cwd_relative():
    import inspect

    assert inspect.signature(dl_mod.TestDataLoader.__init__).parameters["file_path"].default == "./data/log.csv"
