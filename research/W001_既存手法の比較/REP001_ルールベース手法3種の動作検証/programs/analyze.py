"""REP001 の結果を DB から集計して、results/ に表（CSV, summary.json）、fig/ に図を作る。

    .venv/Scripts/python.exe research/W001_.../REP001_.../programs/analyze.py

DB/auctionnet.duckdb を読む（先に `python DB/build_db.py`。run_all.sh が最後に呼ぶ）。
集計の定義は REP001.md「2. 達成とみなす条件」に従う。ここでは、結果を見てから定義を変えない。
"""
import json
import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parents[2]
RES, FIG = HERE / "results", HERE / "fig"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
EPISODES = [0, 1, 2, 3]
POS6 = [0, 8, 16, 24, 32, 40]
COLORS = {"PID": "#1f77b4", "ABid": "#2ca02c", "OnlineLP": "#d62728"}
SWEEPS = [  # (strategy, parameter, default)
    ("PID", "base_action", 15), ("PID", "up_factor", 1.2), ("PID", "down_factor", 0.7),
    ("PID", "low_threshold", 0.7), ("PID", "high_threshold", 1.1),
    ("ABid", "bid_scale", 1.0), ("OnlineLP", "cpa_cap_ratio", 1.5), ("OnlineLP", "episode", 0)]

con = duckdb.connect(str(ROOT / "DB" / "auctionnet.duckdb"), read_only=True)
runs = con.execute("SELECT * FROM runs").df()
runs["spec"] = runs.params_spec.map(lambda s: json.loads(s) if isinstance(s, str) else {})
runs["kw_keys"] = runs.spec.map(lambda s: sorted(s.get("player", {}).get("kwargs", {})))
runs["eps_list"] = runs.spec.map(lambda s: s.get("episodes"))
runs["has_gin"] = runs.spec.map(lambda s: bool(s.get("gin")))
ep = con.execute("SELECT * FROM episodes").df()
ticks = con.execute("SELECT * FROM ticks").df()
agents = con.execute("SELECT * FROM agents").df()
pl = con.execute("SELECT * FROM params_long").df()
out = {}

# ---------------------------------------------------------------- E1 動くか
rep_runs = runs[runs.work == "W001"]
e1 = rep_runs.groupby(["strategy", "status"]).size().unstack(fill_value=0)
out["E1_runs_by_strategy_status"] = e1.to_dict()
out["E1_n_runs"] = int(len(rep_runs))
out["E1_n_error"] = int((rep_runs.status != "ok").sum())
ok = runs[(runs.status == "ok")]
out["E1_all_scores_finite"] = bool(np.isfinite(ok.score).all())
if (rep_runs.status != "ok").any():
    rep_runs[rep_runs.status != "ok"][["run_id", "strategy", "player_index", "sweep_point", "error"]].to_csv(
        RES / "E1_errors.csv", index=False, encoding="utf-8")

# ---------------------------------------------------------------- E2 規模感（実規模の ok run のみ）
full = ok[~ok.has_gin]
e2 = full.groupby("strategy").agg(
    n_runs=("run_id", "count"), n_episodes=("n_episodes", "mean"),
    sim_seconds_mean=("sim_seconds", "mean"), sim_seconds_max=("sim_seconds", "max"),
    peak_rss_mb_mean=("peak_rss_mb", "mean"), peak_rss_mb_max=("peak_rss_mb", "max"),
    bidding_ms_per_call_mean=("bidding_seconds_per_call_mean", lambda x: 1000 * x.mean()),
    bidding_ms_per_call_max=("bidding_seconds_per_call_mean", lambda x: 1000 * x.max()))
e2["sim_seconds_per_episode_mean"] = e2.sim_seconds_mean / e2.n_episodes
e2.to_csv(RES / "E2_scale.csv", encoding="utf-8")
tmax = ticks.merge(runs[["run_id", "strategy"]], on="run_id").groupby("strategy").bidding_seconds.max() * 1000
tmax.to_csv(RES / "E2_bidding_ms_max_single_call.csv", encoding="utf-8")
out["E2"] = e2.round(4).to_dict("index")
out["E2_bidding_ms_max_single_call"] = tmax.round(4).to_dict()

# ---------------------------------------------------------------- E4 3手法の比較（既定設定、48位置×4エピソード）
base = ok[(ok.kw_keys.map(len) == 0) & (~ok.has_gin) & (ok.eps_list.map(lambda x: x == EPISODES))]
base_ep = ep.merge(base[["run_id", "strategy", "player_index"]], on="run_id")
cells = base_ep.pivot_table(index=["player_index", "episode"], columns="strategy", values="score_component")
cells = cells.dropna()
out["E4_n_cells"] = int(len(cells))
tot = cells.sum()
out["E4_total_score_component"] = tot.round(4).to_dict()
out["E4_total_score_normalized_20000"] = (tot / 20000).round(6).to_dict()
out["E4_relative_to_ABid"] = (tot / tot["ABid"]).round(4).to_dict() if "ABid" in tot else None
if {"OnlineLP", "PID", "ABid"} <= set(cells.columns):
    out["E4_match"] = bool(tot["OnlineLP"] > tot["PID"] and tot["OnlineLP"] > tot["ABid"])
    out["E4_cell_win_rate_OnlineLP_vs_PID"] = float((cells.OnlineLP > cells.PID).mean())
    out["E4_cell_win_rate_OnlineLP_vs_ABid"] = float((cells.OnlineLP > cells.ABid).mean())
    out["E4_cell_win_rate_PID_vs_ABid"] = float((cells.PID > cells.ABid).mean())
    out["E4_rank_by_total"] = tot.sort_values(ascending=False).index.tolist()
cells.to_csv(RES / "E4_cells_score_component.csv", encoding="utf-8")
agg = base_ep.groupby("strategy").agg(
    reward_mean=("reward", "mean"), penalty_mean=("penalty", "mean"), penalty_lt1_share=("penalty", lambda x: (x < 1).mean()),
    real_cpa_median=("real_cpa", "median"), cpa_exceedance_median=("cpa_exceedance_rate", "median"), budget_ratio_mean=("budget_consumer_ratio", "mean"),
    win_pv_ratio_mean=("win_pv_ratio", "mean"), score_rank_mean=("score_rank", "mean"),
    score_component_mean=("score_component", "mean"), score_component_median=("score_component", "median"),
    others_score_mean=("others_score_mean", "mean"), last_compete_tick_mean=("last_compete_tick_index", "mean"))
agg.to_csv(RES / "E4_by_strategy.csv", encoding="utf-8")
out["E4_by_strategy"] = agg.round(4).to_dict("index")
base_ep["category"] = base_ep["category"].astype(int)
by_cat = base_ep.groupby(["category", "strategy"]).score_component.sum().unstack()
by_cat.to_csv(RES / "E4_by_category.csv", encoding="utf-8")
by_cat_rank = by_cat.idxmax(axis=1)
out["E4_best_strategy_by_category"] = {int(k): v for k, v in by_cat_rank.items()}
by_ep = base_ep.groupby(["episode", "strategy"]).score_component.sum().unstack()
by_ep.to_csv(RES / "E4_by_episode.csv", encoding="utf-8")
out["E4_best_strategy_by_episode"] = {int(k): v for k, v in by_ep.idxmax(axis=1).items()}
# 位置（セル）のばらつき
out["E4_cell_cv"] = (cells.std() / cells.mean()).round(3).to_dict()

# ---------------------------------------------------------------- E5 掃引
rows = []
for strat, key, default in SWEEPS:
    cand = ok[(ok.strategy == strat) & (~ok.has_gin) & (ok.eps_list.map(lambda x: x == EPISODES))
              & ok.player_index.isin(POS6) & ok.kw_keys.map(lambda k: set(k) <= {key})]
    pv = pl[pl.key == f"player.kwargs.{key}"][["run_id", "value_num"]]
    d = cand.merge(pv, on="run_id").merge(ep, on="run_id")
    for val, g in d.groupby("value_num"):
        n_runs = g.run_id.nunique()
        rows.append(dict(strategy=strat, param=key, value=val, is_default=bool(val == default), n_runs=n_runs,
                         score_component_mean=g.score_component.mean(), reward_mean=g.reward.mean(),
                         penalty_mean=g.penalty.mean(), budget_ratio_mean=g.budget_consumer_ratio.mean(),
                         win_pv_ratio_mean=g.win_pv_ratio.mean(), score_rank_mean=g.score_rank.mean()))
sw = pd.DataFrame(rows)
if len(sw):
    sw["ratio_to_default"] = sw.groupby(["strategy", "param"]).apply(
        lambda g: g.score_component_mean / g.loc[g.is_default, "score_component_mean"].iloc[0]
        if g.is_default.any() else g.score_component_mean * np.nan).reset_index(level=[0, 1], drop=True)
    sw.to_csv(RES / "E5_sweeps.csv", index=False, encoding="utf-8")
    best = sw.loc[sw.groupby(["strategy", "param"]).score_component_mean.idxmax()]
    out["E5_best_per_param"] = best[["strategy", "param", "value", "is_default", "score_component_mean",
                                      "ratio_to_default"]].round(4).to_dict("records")
    out["E5_default_is_best"] = {f"{r.strategy}.{r.param}": bool(r.is_default) for r in best.itertuples()}
    out["E5_n_points"] = int(len(sw))

# ---------------------------------------------------------------- E6 欠損
miss = {}
for name, df in (("episodes", ep), ("ticks", ticks), ("agents", agents)):
    num = df.select_dtypes("number")
    bad = {c: int((~np.isfinite(num[c])).sum()) for c in num.columns if (~np.isfinite(num[c])).any()}
    miss[name] = bad
out["E6_nonfinite_counts"] = miss
out["E6_ok"] = all(len(v) == 0 for v in miss.values())

# ---------------------------------------------------------------- 図
def fig_scores():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    order = [s for s in ["ABid", "PID", "OnlineLP"] if s in cells.columns]
    ax[0].bar(order, [cells[s].sum() / 20000 for s in order], color=[COLORS[s] for s in order])
    ax[0].set_ylabel("score（20000で割った合計）")
    ax[0].set_title("48位置×4エピソード合計")
    parts = ax[1].boxplot([cells[s] for s in order], tick_labels=order, showfliers=True, patch_artist=True)
    for p, s in zip(parts["boxes"], order):
        p.set_facecolor(COLORS[s]); p.set_alpha(0.6)
    ax[1].set_yscale("symlog", linthresh=1)
    ax[1].set_ylabel("セルごとの score_component"); ax[1].set_title("セル(位置×エピソード)のばらつき")
    fig.tight_layout(); fig.savefig(FIG / "fig_scores.png", dpi=200); plt.close(fig)


def fig_alpha():
    t = ticks.merge(base[["run_id", "strategy"]], on="run_id")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for s in ["PID", "ABid", "OnlineLP"]:
        g = t[t.strategy == s].groupby("tick")
        ax[0].plot(g.alpha_eff.median(), color=COLORS[s], label=s)
        ax[1].plot(g.cost.sum() / g.cost.sum().sum() if False else (g.cost.sum().cumsum() / g.cost.sum().sum()), color=COLORS[s], label=s)
        ax[2].plot(g.n_won.mean(), color=COLORS[s], label=s)
    others = t.groupby("tick").alpha_others_median.median()
    ax[0].plot(others, "k--", label="他47人の中央値")
    ax[0].set_yscale("log"); ax[0].set_title("実効入札係数 α（中央値）"); ax[0].set_xlabel("ティック")
    ax[1].set_title("支払いの累積割合"); ax[1].set_xlabel("ティック")
    ax[2].set_title("落札数（平均）"); ax[2].set_xlabel("ティック"); ax[2].set_yscale("symlog", linthresh=1)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig_alpha_time.png", dpi=200); plt.close(fig)


def fig_sweeps():
    if not len(sw):
        return
    groups = list(sw.groupby(["strategy", "param"]))
    n = len(groups)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2))
    for ax, ((s, k), g) in zip(axes.ravel(), groups):
        g = g.sort_values("value")
        ax.plot(g.value.astype(str), g.ratio_to_default, "o-", color=COLORS[s])
        ax.axhline(1.0, color="gray", lw=0.7)
        ax.set_title(f"{s}  {k}", fontsize=9); ax.set_ylabel("既定値に対する成績比", fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig_sweeps.png", dpi=200); plt.close(fig)


def fig_scale():
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    e = e2.reindex([s for s in ["PID", "ABid", "OnlineLP"] if s in e2.index])
    ax[0].bar(e.index, e.sim_seconds_per_episode_mean, color=[COLORS[s] for s in e.index])
    ax[0].set_ylabel("1エピソードの所要時間（秒）"); ax[0].set_title("シミュレーションの所要時間")
    ax[1].bar(e.index, e.bidding_ms_per_call_mean, color=[COLORS[s] for s in e.index])
    ax[1].set_ylabel("bidding 1回（ミリ秒）"); ax[1].set_title("手法の1回の思考時間"); ax[1].set_yscale("log")
    fig.tight_layout(); fig.savefig(FIG / "fig_scale.png", dpi=200); plt.close(fig)


if len(cells):
    fig_scores(); fig_alpha(); fig_scale()
fig_sweeps()
(RES / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
