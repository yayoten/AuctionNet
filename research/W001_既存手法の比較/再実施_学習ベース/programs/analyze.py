"""新 REP003（学習ベース 5 手法の検証）の集計。定義は REP.md 4.1 節（評価の前に決めた。結果を見てから変えない）。

    .venv/bin/python <このREP>/programs/analyze.py

DB/runs/ の原本（meta.json・params.json・parquet）を直接読む（DuckDB の索引は使わない。他のタスクの作り直しと重ならないように）。
使う run：
- 学習ベース：rep = REP003new、spec_name = eval_<手法>_s<seed>、status = ok。
- ルールベース：学習ベースと同じ host・同じ github_version で、既定値（kwargs が空）・エピソード 0〜3・seed 1・gin の上書き無しの run
  （rep は問わない。run_id が rep に依らないので、T003 が流した run も同じもの）。
- 参考（MOPO・COMBO、同梱の重み）：rep = REP003new、spec_name = eval_bundled_mbrl。
出力：results/*.csv、results/summary.json、fig/*.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[2]
RES, FIG = HERE / "results", HERE / "fig"
REP = "REP003new"
RULE = ["PID", "ABid", "OnlineLP"]
LEARNED = ["BC", "IQL", "CQL", "TD3_BC", "BCQ"]
MBRL = ["MOPO", "COMBO"]
EPISODES = [0, 1, 2, 3]
LIMIT = 0.10
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
COLORS = {"PID": "#1f77b4", "ABid": "#2ca02c", "OnlineLP": "#d62728", "BC": "#9467bd", "IQL": "#ff7f0e",
          "CQL": "#8c564b", "BCQ": "#e377c2", "TD3_BC": "#17becf", "MOPO": "#7f7f7f", "COMBO": "#bcbd22"}

# ---------------------------------------------------------------- run を集める
metas = []
for mp in sorted((REPO / "DB" / "runs").glob("R*/meta.json")):
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["_dir"] = mp.parent
    metas.append(m)
mine = [m for m in metas if m.get("rep") == REP and str(m.get("spec_name", "")).startswith("eval_")
        and m.get("strategy") in LEARNED]
assert mine, "REP003new の run がまだ無い"
versions = sorted({m["github_version"] for m in mine})
hosts = sorted({m["host"] for m in mine})
out = {"rep": REP, "github_version": versions, "host": hosts,
       "n_runs_learned_all": len(mine), "n_runs_learned_ok": sum(m["status"] == "ok" for m in mine),
       "n_runs_learned_dirty": sum(bool(m.get("github_dirty")) for m in mine),
       "head_commits": sorted({m["head_commit"] for m in mine})}
assert len(versions) == 1 and len(hosts) == 1, (versions, hosts)
V, HOST = versions[0], hosts[0]


def load(m, method, model_seed=None, group=None):
    par = json.loads((m["_dir"] / "params.json").read_text(encoding="utf-8"))
    min_budget = par["effective"]["env"]["BiddingEnv"]["min_remaining_budget"]
    ep = pd.read_parquet(m["_dir"] / "episodes.parquet")
    tk = pd.read_parquet(m["_dir"] / "ticks.parquet")
    tk["has_budget"] = tk.remaining_budget_before >= min_budget
    rows = []
    for e, g in tk.groupby("episode"):
        hb = g[g.has_budget]
        rows.append(dict(episode=e, alpha_mean=g.alpha_eff.mean(), alpha_std_ticks=g.alpha_eff.std(),
                         alpha_tick0=float(g.alpha_eff[g.tick == 0].iloc[0]),
                         alpha_mean_budget=hb.alpha_eff.mean() if len(hb) else np.nan,
                         alpha_std_ticks_budget=hb.alpha_eff.std() if len(hb) > 1 else np.nan,
                         alpha_min_budget=hb.alpha_eff.min() if len(hb) else np.nan,
                         alpha_max_budget=hb.alpha_eff.max() if len(hb) else np.nan,
                         n_ticks_budget=int(len(hb)), alpha_others_median_mean=g.alpha_others_median.mean(),
                         bidding_ms_max=1000 * g.bidding_seconds.max()))
    ep = ep.merge(pd.DataFrame(rows), on="episode")
    ep = ep.assign(method=method, group=group, model_seed=model_seed, player_index=int(m["player_index"]),
                   sim_seconds=m.get("sim_seconds"), peak_rss_mb=m.get("peak_rss_mb"), model_id=m.get("model_id"))
    tk = tk.assign(method=method, model_seed=model_seed, player_index=int(m["player_index"]))
    return ep, tk[["method", "model_seed", "player_index", "episode", "tick", "alpha_eff", "alpha_others_median", "has_budget",
                   "remaining_budget_before", "n_won", "cost", "reward"]]


eps, tks, seen = [], [], set()
for m in metas:
    if m.get("status") != "ok" or m.get("github_dirty") or m.get("github_version") != V or m.get("host") != HOST:
        continue
    spec = json.loads((m["_dir"] / "params.json").read_text(encoding="utf-8"))["spec"]
    if spec.get("episodes") != EPISODES or spec.get("seed") != 1 or spec.get("gin"):
        continue
    s, kw = m["strategy"], spec["player"].get("kwargs", {})
    if s in LEARNED and m.get("rep") == REP and kw.get("model_dir"):
        mm = json.loads((REPO / kw["model_dir"] / "meta.json").read_text(encoding="utf-8"))
        key, a = (s, mm["seed"], m["player_index"]), load(m, s, mm["seed"], "learned")
    elif s in RULE and not kw:
        key, a = (s, None, m["player_index"]), load(m, s, None, "rule")
    elif s in MBRL and m.get("rep") == REP and not kw:
        key, a = (s, None, m["player_index"]), load(m, s, None, "bundled")
    else:
        continue
    assert key not in seen, key
    seen.add(key)
    eps.append(a[0])
    tks.append(a[1])
cells = pd.concat(eps, ignore_index=True)
ticks = pd.concat(tks, ignore_index=True)
cells["bought"] = cells.reward > 0
cells["exceed"] = cells.bought & (cells.cpa_exceedance_rate > 0)
cells.drop(columns=["run_id"]).to_csv(RES / "cells.csv", index=False)
num = cells.select_dtypes("number").drop(columns=["model_seed", "alpha_mean_budget", "alpha_std_ticks_budget", "alpha_min_budget",
                                                  "alpha_max_budget"], errors="ignore")
out["n_nonfinite_in_cells"] = {c: int((~np.isfinite(num[c])).sum()) for c in num if (~np.isfinite(num[c])).any()}
out["n_cells_by_method_seed"] = {f"{k[0]}/{k[1]}": int(v) for k, v in
                                 cells.fillna({"model_seed": 0}).groupby(["method", "model_seed"]).size().items()}


# ---------------------------------------------------------------- V3・V5 手法 × seed
def agg(g):
    b = g[g.bought]
    return pd.Series(dict(
        n_cells=len(g), score_mean=g.score_component.mean(), score_sum=g.score_component.sum(), reward_mean=g.reward.mean(),
        budget_ratio_mean=g.budget_consumer_ratio.mean(), budget_ratio_min=g.budget_consumer_ratio.min(),
        n_cells_no_spend=int((g.budget_consumer_ratio == 0).sum()), n_cells_no_purchase=int((~g.bought).sum()),
        exceed_share_of_bought=b.exceed.mean() if len(b) else np.nan,
        exceed_rate_median_of_bought=b.cpa_exceedance_rate.median() if len(b) else np.nan,
        penalized_share_all=(g.penalty < 1).mean(),
        reward_lost_share=1 - g.score_component.sum() / g.reward.sum() if g.reward.sum() else np.nan,
        rank_median=g.score_rank.median(), others_score_mean=g.others_score_mean.mean(),
        alpha_mean=g.alpha_mean.mean(), alpha_mean_budget=g.alpha_mean_budget.mean(),
        alpha_std_ticks=g.alpha_std_ticks.mean(), alpha_std_ticks_budget=g.alpha_std_ticks_budget.mean(),
        alpha_tick0_mean=g.alpha_tick0.mean(), alpha_tick0_std_cells=g.alpha_tick0.std(),
        alpha_min_budget=g.alpha_min_budget.min(), alpha_max_budget=g.alpha_max_budget.max(),
        alpha_others_median_mean=g.alpha_others_median_mean.mean(),
        sim_seconds_mean=g.sim_seconds.mean(), peak_rss_mb_max=g.peak_rss_mb.max(), bidding_ms_max=g.bidding_ms_max.max()))


by_seed = cells[cells.group == "learned"].groupby(["method", "model_seed"]).apply(agg).reset_index()
by_seed["method"] = pd.Categorical(by_seed.method, LEARNED)
by_seed = by_seed.sort_values(["method", "model_seed"])
by_seed.to_csv(RES / "by_seed.csv", index=False)
by_method = cells.groupby("method").apply(agg)
by_method = by_method.loc[[m for m in RULE + LEARNED + MBRL if m in by_method.index]]
spread = {}
for a, g in by_seed.groupby("method", observed=True):
    mean = g.score_mean.mean()
    spread[a] = dict(n_seeds=int(len(g)), score_mean_by_seed=[round(float(x), 3) for x in g.score_mean], score_mean_3seeds=round(float(mean), 3),
                     spread=round(float((g.score_mean.max() - g.score_mean.min()) / mean), 4) if mean else None,
                     within_10pct=bool(mean and (g.score_mean.max() - g.score_mean.min()) / mean <= LIMIT),
                     complete=bool(len(g) == 3 and (g.n_cells == 192).all()),
                     n_cells_no_spend=int(g.n_cells_no_spend.sum()), n_cells_no_purchase=int(g.n_cells_no_purchase.sum()))
    by_method.loc[a, "seed_spread"] = spread[a]["spread"]
by_method.to_csv(RES / "by_method.csv")
out["V3_by_method"] = spread

# ---------------------------------------------------------------- V5 BCQ の性質
bq = cells[cells.method == "BCQ"]
if bq.model_seed.nunique() == 3:
    piv = bq.pivot_table(index=["player_index", "episode"], columns="model_seed", values=["score_component", "reward", "all_cost"])
    same = all(bool((piv[c].nunique(axis=1) == 1).all()) for c in ("score_component", "reward", "all_cost"))
    tb = ticks[ticks.method == "BCQ"]
    hb = tb[tb.has_budget]
    out["V5_BCQ"] = dict(three_seeds_identical_all_cells=same, n_ticks=int(len(tb)), n_ticks_with_budget=int(len(hb)),
                         n_ticks_with_budget_alpha_100=int(np.isclose(hb.alpha_eff, 100, atol=1e-3).sum()),
                         n_ticks_alpha_100=int(np.isclose(tb.alpha_eff, 100, atol=1e-3).sum()),
                         n_ticks_alpha_0=int(np.isclose(tb.alpha_eff, 0, atol=1e-9).sum()),
                         alpha_min_with_budget=float(hb.alpha_eff.min()), alpha_max_with_budget=float(hb.alpha_eff.max()))
at = (ticks.groupby(["method", "tick"]).agg(alpha_median=("alpha_eff", "median"), alpha_q25=("alpha_eff", lambda x: x.quantile(.25)),
                                             alpha_q75=("alpha_eff", lambda x: x.quantile(.75)), has_budget_share=("has_budget", "mean"),
                                             alpha_others_median=("alpha_others_median", "median")).reset_index())
at.to_csv(RES / "alpha_by_tick.csv", index=False)

# ---------------------------------------------------------------- V4 8 手法の比較（学習ベースは、セルごとに 3 seed の平均）
cm = cells.groupby(["method", "player_index", "episode"]).agg(score=("score_component", "mean"), n=("score_component", "size")).reset_index()
wide = cm.pivot_table(index=["player_index", "episode"], columns="method", values="score")
wide = wide[[m for m in RULE + LEARNED + MBRL if m in wide.columns]]
wide.to_csv(RES / "cells_score_by_method.csv")
out["V4_n_cells_per_method"] = wide.notna().sum().astype(int).to_dict()
eight = [m for m in RULE + LEARNED if m in wide.columns]
full = wide[eight].dropna()
out["V4_n_cells_common_8"] = int(len(full))
out["V4_complete"] = bool(len(eight) == 8 and len(full) == 192)
allm = wide.loc[full.index] if len(full) else wide
tab = pd.DataFrame({"score_sum": allm.sum(min_count=1), "score_mean": allm.mean(), "n_cells": allm.notna().sum()})
tab["score_normalized_20000"] = tab.score_sum / 20000
if "ABid" in tab.index:
    tab["rel_ABid"] = tab.score_sum / tab.score_sum["ABid"]
if "OnlineLP" in tab.index:
    tab["rel_OnlineLP"] = tab.score_sum / tab.score_sum["OnlineLP"]
    tab["share_cells_above_OnlineLP"] = [(allm[m] > allm["OnlineLP"]).mean() if m != "OnlineLP" else np.nan for m in tab.index]
if len(eight) == 8 and len(full):
    best = full.idxmax(axis=1).value_counts(normalize=True)
    tab["share_cells_best_of_8"] = [best.get(m, 0.0) if m in eight else np.nan for m in tab.index]
cols = ["reward_mean", "budget_ratio_mean", "budget_ratio_min", "n_cells_no_purchase", "exceed_share_of_bought", "exceed_rate_median_of_bought",
        "penalized_share_all", "reward_lost_share", "rank_median", "alpha_mean", "alpha_mean_budget", "alpha_std_ticks_budget",
        "alpha_tick0_mean", "alpha_tick0_std_cells", "others_score_mean", "seed_spread", "sim_seconds_mean", "peak_rss_mb_max", "bidding_ms_max"]
tab = tab.join(by_method[[c for c in cols if c in by_method]])
tab["group"] = ["rule" if m in RULE else "learned" if m in LEARNED else "bundled" for m in tab.index]
tab.to_csv(RES / "table_methods.csv")
out["V4_order_by_score_sum_8"] = list(tab.loc[eight].sort_values("score_sum", ascending=False).index) if len(eight) == 8 else None
out["V4_table"] = json.loads(tab.round(4).to_json(orient="index"))
(RES / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- 図
order = list(tab.index)
fig, ax = plt.subplots(figsize=(7.2, 3.0))
x = np.arange(len(order))
ax.bar(x, tab.score_mean, color=[COLORS[m] for m in order], alpha=0.85)
for i, m in enumerate(order):
    g = by_seed[by_seed.method == m]
    if len(g):
        ax.scatter([i] * len(g), g.score_mean, color="k", s=14, zorder=3, label="seed ごとの平均" if m == LEARNED[0] else None)
ax.set_xticks(x)
ax.set_xticklabels([m + ("\n(同梱)" if m in MBRL else "") for m in order])
ax.set_ylabel("192 セルの平均 score")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig_scores.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), sharey=True)
for ax, group, title in ((axes[0], RULE, "ルールベース"), (axes[1], LEARNED + MBRL, "学習ベース（MOPO・COMBO は同梱の重み）")):
    for m in group:
        g = at[at.method == m]
        if len(g):
            ax.plot(g.tick, g.alpha_median, color=COLORS[m], label=m, lw=1.4, ls="--" if m in MBRL else "-")
    g = at[at.method == order[0]]
    ax.plot(g.tick, g.alpha_others_median, color="k", lw=0.8, ls=":", label="他の 47 社の中央値")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("ティック")
    ax.legend(frameon=False, fontsize=7, ncol=2)
axes[0].set_ylabel("実効の α の中央値")
fig.tight_layout()
fig.savefig(FIG / "fig_alpha_time.png", dpi=200)
plt.close(fig)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(by_seed.round(3).to_string())
print(tab.round(3).T.to_string())
print(json.dumps({k: v for k, v in out.items() if k != "V4_table"}, ensure_ascii=False, indent=1))
