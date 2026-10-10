"""05: 新しいコード（記録の形式の版 2）で流した run が、同じ設定の既存の run（版 1）と、全セルで一致すること。

指示書（T006）の 3：「記録を足しても、結果を変えない」ことの、本番の規模（PV 数 50 万）での確認である。
相手は、再実施（T003）の `check_same_as_old` の 18 run（ルールベース 3 手法 × 6 位置 × エピソード 0〜3 = 72 セル）。
新しい側は、`params/check_same_as_old_v2.json` を `research/src/run_experiment.py` で流したもの（DB/runs/ に入る）。
どちらかが DB/runs/ に無い端末では、理由を表示してスキップする。
"""
import json

import pandas as pd
import pytest

import build_db
from conftest import DB_DIR, HERE

pytestmark = pytest.mark.needs_db
RUNS = DB_DIR / "runs"
EP_COLS = ["reward", "all_cost", "real_cpa", "penalty", "score_component", "all_win_pv", "budget_consumer_ratio",
           "second_price_ratio", "cpa_exceedance_rate", "last_compete_tick_index", "bid_mean", "score_rank", "others_score_mean"]
TICK_COLS = ["num_pv", "pvalue_mean", "bid_mean", "alpha_eff", "n_won", "n_slot1", "n_slot2", "n_slot3", "n_exposed", "reward",
             "cost", "remaining_budget_before", "lwc_mean", "total_cost_all", "n_active_agents"]
AGENT_COLS = ["reward", "cost", "final_remaining_budget", "score_component"]


def load_pairs():
    """(設定のハッシュ, 位置) → {"old": フォルダ, "new": フォルダ}。old = 版 1、new = 版 2 の、同じ設定の run。"""
    spec = json.loads((HERE / "params" / "check_same_as_old_v2.json").read_text(encoding="utf-8"))
    want_strategies, want_pos = set(spec["sweep"]["player.strategy"]), set(spec["player_indices"])
    pairs = {}
    for d in sorted(RUNS.iterdir()) if RUNS.is_dir() else []:
        mp, pp = d / "meta.json", d / "params.json"
        if not (mp.exists() and pp.exists()):
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m.get("status") != "ok" or m.get("github_dirty") or m.get("strategy") not in want_strategies \
                or m.get("player_index") not in want_pos or not m.get("os", "").startswith("Linux"):
            continue
        s = json.loads(pp.read_text(encoding="utf-8"))["spec"]
        if s.get("player", {}).get("kwargs") or s.get("gin") or s.get("episodes") != spec["episodes"] or s.get("seed") != spec["seed"]:
            continue
        side = "new" if m.get("record_version", 1) >= 2 else "old"
        cur = pairs.setdefault((build_db.config_key(s), m["player_index"]), {})
        if side not in cur or m["created_at"] > cur[side][1]:
            cur[side] = (d, m["created_at"])
    return {k: {s: v[0] for s, v in p.items()} for k, p in pairs.items() if len(p) == 2}


PAIRS = load_pairs()


def test_all_18_runs_have_a_counterpart():
    if not PAIRS:
        pytest.skip("DB/runs/ に、照合する run の組が無い（check_same_as_old_v2.json を流していない、または版 1 の run が無い）")
    assert len(PAIRS) == 18, f"組になった run が {len(PAIRS)} 個（18 個のはず）"


@pytest.mark.parametrize("table,cols,key", [("episodes", EP_COLS, ["episode"]), ("ticks", TICK_COLS, ["episode", "tick"]),
                                            ("agents", AGENT_COLS, ["episode", "agent_index"])])
def test_every_cell_is_identical(table, cols, key):
    if not PAIRS:
        pytest.skip("DB/runs/ に、照合する run の組が無い")
    n = 0
    for (cfg, pos), p in PAIRS.items():
        old = pd.read_parquet(p["old"] / f"{table}.parquet").sort_values(key).reset_index(drop=True)
        new = pd.read_parquet(p["new"] / f"{table}.parquet").sort_values(key).reset_index(drop=True)
        assert len(old) == len(new)
        for c in cols:
            assert old[c].tolist() == new[c].tolist(), (p["old"].name, p["new"].name, table, c)       # 丸めの差も許さない
        n += len(old)
    assert n == {"episodes": 72, "ticks": 72 * 48, "agents": 72 * 48}[table]
