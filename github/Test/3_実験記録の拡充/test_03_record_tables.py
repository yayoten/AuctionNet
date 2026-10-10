"""03: 記録の中身（表・列・段）と、表どうしの整合。

- 記録の段（basic / standard / detail / full）ごとに、決めたファイルができる。
- ティックの合計 = エピソードの値（REP006.md の S4）。全 48 社の表・機会ごとの表も、同じ値になる。
- REP006 の 3 量（N_est・N_real・Σp(1−p)）から、恒等式 R = E × P × L が成り立つ（S3）。
- 手法の内部状態が残る。
"""
import json

import numpy as np
import pandas as pd
import pytest

import run_experiment as rx
from conftest import run_one

pytestmark = pytest.mark.slow
BASIC = {"params.json", "episodes.parquet", "ticks.parquet", "agents.parquet", "sim_iters.parquet"}
STANDARD = BASIC | {"raw/agent_ticks.parquet", "raw/pv_won.parquet"}
P = 3


def read(d, name):
    return pd.read_parquet(d / name)


# ---------- 段とファイル ----------
def test_record_levels_are_ordered_and_default_is_standard(tmp_path):
    assert rx.RECORD_LEVELS == {"basic": 0, "standard": 1, "detail": 2, "full": 3} and rx.RECORD_VERSION == 2
    sp = tmp_path / "s.json"
    sp.write_text(json.dumps(dict(name="t", player=dict(strategy="PID"), player_indices=[0], episodes=[0])), encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rx, "DB_RUNS", tmp_path / "runs")
        (task,), _ = rx.build_tasks(str(sp), False)
        assert task["resolved"]["record"] == "standard"
        (t2,), _ = rx.build_tasks(str(sp), False, record="full")
        assert t2["resolved"]["record"] == "full" and t2["run_id"] == task["run_id"]


def test_unknown_record_level_is_rejected(tmp_path):
    sp = tmp_path / "s.json"
    sp.write_text(json.dumps(dict(name="t", player=dict(strategy="PID"), record="everything")), encoding="utf-8")
    with pytest.raises(AssertionError):
        rx.build_tasks(str(sp), False)


@pytest.mark.parametrize("record,want", [("basic", BASIC), ("standard", STANDARD), ("detail", STANDARD | {"raw/pv_all.parquet"})])
def test_files_of_each_level(record, want, tmp_path):
    m, d = run_one(tmp_path, "PID", record, episodes=(0,))
    files = json.loads(m["files"])
    assert set(files) == want
    assert m["record_level"] == record and m["record_version"] == 2
    assert m["bytes_total"] == sum(files.values()) and m["bytes_raw"] == sum(v for k, v in files.items() if k.startswith("raw/"))


def test_full_level_adds_all_bids(full_runs):
    m, d = full_runs["PID"]
    assert set(json.loads(m["files"])) == STANDARD | {"raw/pv_all.parquet", "raw/bids_all_ep0.parquet", "raw/bids_all_ep1.parquet"}
    b, a = read(d, "raw/bids_all_ep0.parquet"), read(d, "raw/pv_all.parquet")
    a0 = a[a.episode == 0]
    assert len(b) == len(a0) and [c for c in b.columns if c.startswith("bid_")] == [f"bid_{i:02d}" for i in range(48)]
    assert np.array_equal(b[f"bid_{P:02d}"].to_numpy(), a0.bid.to_numpy())              # プレイヤーの列 = 機会ごとの記録の入札額
    assert np.array_equal(b.winner1.to_numpy(), a0.winner1.to_numpy())


def test_a_lower_level_run_is_rerun_and_a_sufficient_one_is_skipped(tmp_path):
    m, d = run_one(tmp_path, "PID", "basic", episodes=(0,))
    sp = tmp_path / "spec_PID_basic_3.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rx, "DB_RUNS", tmp_path / "runs")
        tasks, skipped = rx.build_tasks(str(sp), False)
        assert (len(tasks), skipped) == (0, [m["run_id"]])                                # 同じ段なら、スキップ
        tasks, skipped = rx.build_tasks(str(sp), False, record="standard")
        assert [t["run_id"] for t in tasks] == [m["run_id"]] and skipped == []           # 段を上げると、流し直す
        rx.execute_run(tasks[0])
        tasks, skipped = rx.build_tasks(str(sp), False, record="basic")
        assert tasks == [] and skipped == [m["run_id"]]                                   # 高い段があれば、低い段の要求はスキップ


# ---------- 整合（S4） ----------
@pytest.mark.parametrize("strategy", ["PID", "ABid", "OnlineLP", "IQL"])
def test_tick_sums_equal_episode_values(strategy, full_runs):
    m, d = full_runs[strategy]
    e, t, a = read(d, "episodes.parquet").set_index("episode"), read(d, "ticks.parquet"), read(d, "agents.parquet")
    g = t.groupby("episode").sum(numeric_only=True)
    assert len(t) == 96 and len(a) == 96 and len(e) == 2
    np.testing.assert_allclose(g.cost, e.all_cost, rtol=1e-12)
    assert g.reward.tolist() == e.reward.tolist()
    assert g.n_exposed.tolist() == e.all_win_pv.tolist() and g.n_won.tolist() == e.n_won.tolist()
    for tick_col, ep_col in [("est_sum_exposed", "n_est"), ("real_sum_exposed", "n_real"), ("real_var_exposed", "real_var")]:
        np.testing.assert_allclose(g[tick_col], e[ep_col], rtol=1e-12)
    np.testing.assert_allclose(g.cost_slot1 + g.cost_slot2 + g.cost_slot3, g.cost, rtol=1e-9, atol=1e-9)
    assert (t.n_exposed_slot1 + t.n_exposed_slot2 + t.n_exposed_slot3 == t.n_exposed).all()
    assert (t.reward_slot1 + t.reward_slot2 + t.reward_slot3 == t.reward).all()
    mine = a[a.is_player].set_index("episode")
    for col in ["n_est", "n_real", "real_var", "trunc_u1", "trunc_u2", "budget_exhausted_tick", "n_bids_dropped"]:
        assert mine[col].tolist() == e[col].tolist(), col
    assert (t.real_sum_exposed <= t.real_sum_won + 1e-12).all() and (t.real_sum_won <= t.real_sum_all + 1e-12).all()
    assert (t.n_sim_iters >= 1).all() and g.n_sim_iters.tolist() == e.n_sim_iters.tolist()


def test_agent_ticks_sum_to_agents_and_player_row_equals_ticks(full_runs):
    m, d = full_runs["PID"]
    at, a, t = read(d, "raw/agent_ticks.parquet"), read(d, "agents.parquet"), read(d, "ticks.parquet")
    assert len(at) == 2 * 48 * 48 and at.groupby(["episode", "tick"]).is_player.sum().eq(1).all()
    g = at.groupby(["episode", "agent_index"]).sum(numeric_only=True)
    a = a.set_index(["episode", "agent_index"]).loc[g.index]
    np.testing.assert_allclose(g.cost, a.cost, rtol=1e-12)
    assert g.reward.tolist() == a.reward.tolist()
    for x, y in [("est_sum_exposed", "n_est"), ("real_sum_exposed", "n_real"), ("real_var_exposed", "real_var")]:
        np.testing.assert_allclose(g[x], a[y], rtol=1e-12)
    pl = at[at.is_player].sort_values(["episode", "tick"]).reset_index(drop=True)
    assert (pl.agent_index == P).all()
    for col in ["cost", "reward", "alpha_eff", "n_won", "n_exposed", "est_sum_exposed", "real_sum_exposed", "real_var_exposed",
                "bid_p50", "n_bids_dropped", "remaining_budget_before", "remaining_budget_after"]:
        np.testing.assert_allclose(pl[col].to_numpy(float), t[col].to_numpy(float), rtol=1e-12, err_msg=col)
    # 1 つの機会で露出できるのは 3 社まで。全社の露出の合計 = ticks.n_exposed_all
    assert at.groupby(["episode", "tick"]).n_exposed.sum().tolist() == t.n_exposed_all.tolist()
    # 残り予算のつながり：あとの残り = 前の残り − 支払い
    np.testing.assert_allclose(at.remaining_budget_after, at.remaining_budget_before - at.cost, atol=1e-9)


def test_pv_all_reproduces_tick_values_and_pv_won_is_its_subset(full_runs):
    m, d = full_runs["PID"]
    al, w, t = read(d, "raw/pv_all.parquet"), read(d, "raw/pv_won.parquet"), read(d, "ticks.parquet")
    assert len(al) == t.num_pv.sum()
    assert al.groupby(["episode", "tick"]).pv_index.apply(lambda s: s.tolist() == list(range(len(s)))).all()
    ex = al[al.is_exposed == 1]
    g = ex.groupby(["episode", "tick"]).agg(cost=("price", "sum"), reward=("conversion", "sum"), est=("pvalue", "sum"), n=("price", "size"))
    tt = t.set_index(["episode", "tick"]).loc[g.index]
    np.testing.assert_allclose(g.cost, tt.cost, rtol=1e-5)                               # 機会ごとの記録は float32
    np.testing.assert_allclose(g.est, tt.est_sum_exposed, rtol=1e-5)
    assert g.reward.tolist() == tt.reward.astype(int).tolist() and g.n.tolist() == tt.n_exposed.tolist()
    np.testing.assert_allclose(np.clip(ex.value_real, 0, 1).groupby([ex.episode, ex.tick]).sum(), tt.real_sum_exposed, rtol=1e-5)
    # 行の中の整合
    assert (al.conversion == al.conversion_draw * al.is_exposed).all()
    assert (al.is_exposed <= (al.slot > 0)).all() and (al.is_exposed <= al.exposure_draw).all()
    won = al[al.slot > 0]
    assert (won.bid_rank == won.slot).all()                                              # 落札した機会では、順位 = 枠
    assert (won.price <= won.bid + 1e-6).all()                                           # 第二価格：払うのは、自分の入札額以下
    for k, col in [(1, "top2_bid"), (2, "top3_bid"), (3, "top4_bid")]:
        s = won[won.slot == k]
        np.testing.assert_allclose(s.price, s[col], rtol=1e-6)                           # 支払い単価 = 1 つ下の順位の入札額
        assert (s[f"winner{k}"] == P).all()
    # pv_won = 落札した機会 ＋ 取り消された入札
    key = ["episode", "tick", "pv_index"]
    want = al[(al.slot > 0) | ((al.bid_before_adjust > 0) & (al.bid == 0))]
    pd.testing.assert_frame_equal(w.reset_index(drop=True), want.reset_index(drop=True))
    assert not w.duplicated(key).any()


# ---------- REP006 の恒等式（S3） ----------
@pytest.mark.parametrize("strategy", ["PID", "ABid"])
def test_epl_identity_from_recorded_quantities(strategy, full_runs):
    m, d = full_runs[strategy]
    a = read(d, "agents.parquet")
    a = a[(a.reward >= 1) & (a.n_est > 0)]
    assert len(a) >= 5, "購入のある広告主が少なすぎて、恒等式を確かめられない"
    E, Pp, L = a.cost / a.n_est / a.cpa_constraint, a.n_est / a.n_real, a.n_real / a.reward
    R = a.cost / a.reward / a.cpa_constraint
    assert np.abs(np.log(E * Pp * L) - np.log(R)).max() <= 1e-9
    assert (a.real_var > 0).all() and (a.real_var <= a.n_real).all()                     # Σp(1−p) ≤ Σp


# ---------- 手法の内部状態 ----------
def test_internal_state_of_each_strategy_is_recorded(full_runs):
    for s, keys in [("PID", {"alpha", "branch", "last_tick_cost", "low_ratio", "high_ratio"}),
                    ("ABid", {"alpha", "base_action", "pvalue_mean"}),
                    ("OnlineLP", {"table_rows", "table_cpa", "alpha_before_cap", "alpha", "capped"}), ("IQL", {"alpha"})]:
        m, d = full_runs[s]
        at = read(d, "raw/agent_ticks.parquet")
        pl = at[at.is_player & at.did_bid]
        assert len(pl) > 0 and all(set(json.loads(j)) == keys for j in pl.internal_json), s
        assert pl.alpha_internal.notna().all()
        if s == "IQL":
            assert all(len(x) == 16 for x in pl.state_raw) and all(len(x) == 16 for x in pl.state_norm)
            assert pl.state_raw.iloc[0][0] == 1.0                                         # 最初のティックの time_left
        else:
            assert pl.state_raw.isna().all()
        if s != "ABid":      # 線形に入札する手法では、取り消しが無ければ、内部の α = 実効の α
            ok = pl[pl.n_bids_dropped == 0]
            np.testing.assert_allclose(ok.alpha_internal, ok.alpha_eff, rtol=1e-6)


def test_pid_branch_matches_alpha_path(full_runs):
    m, d = full_runs["PID"]
    at = read(d, "raw/agent_ticks.parquet")
    pl = at[at.is_player & at.did_bid & (at.episode == 0)].sort_values("tick")
    j = [json.loads(x) for x in pl.internal_json]
    assert j[0]["branch"] == "init" and j[0]["alpha"] == 15
    factor = {"up": 1.2, "down": 0.7, "keep": 1.0}
    for prev, cur in zip(j, j[1:]):
        assert cur["alpha"] == pytest.approx(prev["alpha"] * factor[cur["branch"]], rel=1e-12)
    assert {x["branch"] for x in j[1:]} <= set(factor)


def test_background_agents_have_internal_state_too(full_runs):
    m, d = full_runs["PID"]
    at, a = read(d, "raw/agent_ticks.parquet"), read(d, "agents.parquet")
    cls = a[a.episode == 0].set_index("agent_index").strategy_class
    assert cls[P] == "PidBiddingStrategy" and cls.nunique() >= 8
    first = at[(at.episode == 0) & (at.tick == 0)].set_index("agent_index")
    assert first.did_bid.all() and first.internal_json.notna().all()
    learned = [i for i in cls.index if cls[i] not in ("PidBiddingStrategy", "OnlineLpBiddingStrategy")]
    assert first.loc[learned].state_raw.notna().all()


# ---------- 時間・資源・再現 ----------
def test_meta_has_time_resource_and_reproducibility_fields(full_runs):
    m, d = full_runs["IQL"]
    saved = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    for k in ["record_level", "record_version", "started_at_epoch", "ended_at", "pid", "cpu_model", "machine", "loadavg1_start",
              "env_threads", "torch_num_threads", "run_experiment_sha1", "lib_versions", "import_seconds",
              "strategy_init_seconds", "write_seconds", "hook_seconds_total", "sim_cpu_seconds", "cpu_seconds",
              "cpu_user_seconds", "cpu_system_seconds", "bidding_seconds_all_total", "env_sim_seconds_total",
              "adjust_seconds_total", "n_sim_iters_total", "maxrss_mb", "mem_total_mb", "mem_available_start_mb",
              "gin_file_sha1", "gin_config", "gin_operative_config", "seeds", "files", "bytes_total", "bytes_raw"]:
        assert k in saved and saved[k] is not None, k
    seeds = json.loads(saved["seeds"])
    assert seeds["env_default_seed"] == 1 and seeds["env_conversion_seed"] == 2 and seeds["episodes"] == [0, 1]
    assert "PVNUM = 6000" in saved["gin_config"] and json.loads(saved["lib_versions"])["scipy"]
    assert saved["maxrss_mb"] >= saved["peak_rss_mb"] * 0.9 and saved["n_sim_iters_total"] >= 96
    assert saved["sim_seconds"] >= saved["env_sim_seconds_total"] > 0


def test_tick_time_breakdown_is_consistent(full_runs):
    m, d = full_runs["PID"]
    t, it = read(d, "ticks.parquet"), read(d, "sim_iters.parquet")
    env = t[[c for c in t.columns if c.startswith("env_") and c.endswith("_seconds")]]
    assert env.shape[1] == 7 and (env >= 0).all().all()
    assert (env.sum(axis=1) <= t.sim_seconds + 1e-6).all()                               # 環境の各段の和 ≤ 環境の呼び出しの時間
    assert (t.bidding_seconds <= t.bidding_seconds_all + 1e-9).all()
    assert (t.bidding_seconds_all + t.sim_seconds + t.adjust_seconds <= t.tick_core_seconds + 1e-6).all()
    assert (t.hook_seconds > 0).all() and t.ended_at_epoch.is_monotonic_increasing
    # sim_iters：ティックごとの行数 = n_sim_iters。最後の回は、超過が無い
    n = it.groupby(["episode", "tick"]).size()
    assert n.tolist() == t.n_sim_iters.tolist()
    last = it.sort_values("iter").groupby(["episode", "tick"]).tail(1)
    assert (last.over_cost_ratio_max == 0).all() and (last.n_overcost_agents == 0).all()
    np.testing.assert_allclose(last.sort_values(["episode", "tick"]).player_cost, t.cost, rtol=1e-12)
    first = it[it.iter == 0].sort_values(["episode", "tick"])
    np.testing.assert_allclose(first.player_cost, t.cost_first_iter, rtol=1e-12)
