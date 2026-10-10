"""新 REP002 の集計。results/ に表（CSV, summary.json）、fig/ に図を作る。

    .venv/bin/python research/W001_既存手法の比較/再実施_ルールベース/programs/analyze.py --use old|new

--use old：Linux の既存の 144 run（github の版 0536c690ef）を結果として使う（判定が「一致」のとき）
--use new：今の版で流し直した 144 run（rep=REP002new、spec_name=eval_rule_based_new ほか）を使う
集計の定義は REP.md の 2.（実験の前に決めた）。ここでは、結果を見てから定義を変えない。
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import HERE, OLD_VERSION, STRATS, default_rule_based, load_metas, read_table

ap = argparse.ArgumentParser()
ap.add_argument("--use", choices=["old", "new"], required=True)
args = ap.parse_args()
RES, FIG = HERE / "results", HERE / "fig"
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
COLORS = {"PID": "#1f77b4", "ABid": "#2ca02c", "OnlineLP": "#d62728"}
LABEL = {"PID": "PID", "ABid": "ABid", "OnlineLP": "Online LP"}

allm = default_rule_based(load_metas())
linux = allm[allm.os.str.startswith("Linux")]
if args.use == "old":
    runs = linux[linux.github_version == OLD_VERSION]
else:
    runs = linux[(linux.rep == "REP002new") & (linux.spec_name != "repro_pid_p0")]
runs = runs.sort_values(["strategy", "player_index"]).reset_index(drop=True)
assert not runs.duplicated(["strategy", "player_index"]).any()
out = {"use": args.use}

# ---------------------------------------------------------------- V1 動くか
out["V1"] = dict(n_runs=int(len(runs)), by_strategy=runs.strategy.value_counts().to_dict(),
                 status=runs.status.value_counts().to_dict(), n_dirty=int(runs.github_dirty.sum()),
                 github_versions=sorted(runs.github_version.unique()), hosts=sorted(runs.host.unique()),
                 os=sorted(runs.os.unique()), reps=runs.rep.value_counts().to_dict(),
                 spec_names=runs.spec_name.value_counts().to_dict(), head_commits=sorted(runs.head_commit.unique()),
                 python=sorted(runs.python.unique()), torch=sorted(runs.torch.unique()), numpy=sorted(runs.numpy.unique()),
                 all_scores_finite=bool(np.isfinite(runs.score).all()),
                 created=[runs.created_at.min(), runs.created_at.max()])
out["V1_ok"] = bool(len(runs) == 144 and (runs.status == "ok").all() and not runs.github_dirty.any()
                    and runs.github_version.nunique() == 1 and runs.host.nunique() == 1 and np.isfinite(runs.score).all()
                    and (runs.strategy.value_counts() == 48).all())
runs[["run_id", "strategy", "player_index", "rep", "spec_name", "github_version", "head_commit", "created_at", "status",
      "score"]].to_csv(RES / "V1_runs_used.csv", index=False)

ep = read_table(runs.run_id, "episodes").merge(runs[["run_id", "strategy", "player_index"]], on="run_id")
ticks = read_table(runs.run_id, "ticks").merge(runs[["run_id", "strategy", "player_index"]], on="run_id")
agents = read_table(runs.run_id, "agents")


# ---------------------------------------------------------------- V2 規模感
def scale(m, t):
    g = m.groupby("strategy")
    d = pd.DataFrame(dict(n_runs=g.size(), sim_seconds_mean=g.sim_seconds.mean(), sim_seconds_max=g.sim_seconds.max(),
                          peak_rss_mb_mean=g.peak_rss_mb.mean(), peak_rss_mb_max=g.peak_rss_mb.max(),
                          bidding_ms_mean=g.bidding_seconds_per_call_mean.mean() * 1000))
    d["bidding_ms_max_single_call"] = t.groupby("strategy").bidding_seconds.max() * 1000
    return d.loc[STRATS]


v2 = scale(runs, ticks)
v2.to_csv(RES / "V2_scale.csv")
out["V2"] = v2.round(3).to_dict("index")
new18 = linux[(linux.rep == "REP002new") & (linux.spec_name == "check_same_as_old") & (linux.status == "ok")]   # 再実行した PID・位置 0 は除く（並列数が違う）
if len(new18):
    t18 = read_table(new18.run_id, "ticks").merge(new18[["run_id", "strategy"]], on="run_id")
    v2n = scale(new18, t18)
    v2n.to_csv(RES / "V2_scale_new18_3workers.csv")
    out["V2_new18_3workers"] = v2n.round(3).to_dict("index")

# ---------------------------------------------------------------- V4 3 手法の比較
cells = ep.pivot_table(index=["player_index", "episode"], columns="strategy", values="score_component")[STRATS]
assert cells.shape == (192, 3) and cells.notna().all().all()
cells.to_csv(RES / "V4_cells_score_component.csv")
g = ep.groupby("strategy")
v4 = pd.DataFrame(dict(
    n_cells=g.size(), score_sum=g.score_component.sum(), score_sum_div20000=g.score_component.sum() / 20000,
    score_cell_mean=g.score_component.mean(), score_cell_cv=g.score_component.std() / g.score_component.mean(),
    reward_mean=g.reward.mean(), penalty_mean=g.penalty.mean(),
    budget_ratio_mean=g.budget_consumer_ratio.mean(), budget_ratio_lt09_share=g.budget_consumer_ratio.apply(lambda x: (x < 0.9).mean()),
    stopped_before_last_tick_share=g.last_compete_tick_index.apply(lambda x: (x < 47).mean()),
    rank_median=g.score_rank.median(), rank_top10_share=g.score_rank.apply(lambda x: (x <= 10).mean()),
    others_score_mean=g.others_score_mean.mean())).loc[STRATS]
v4["rel_to_ABid"] = v4.score_sum / v4.score_sum["ABid"]
v4.to_csv(RES / "V4_by_strategy.csv")
out["V4"] = v4.round(4).to_dict("index")
out["V4_pairwise_win_share"] = {f"{a}>{b}": round(float((cells[a] > cells[b]).mean()), 4) for a in STRATS for b in STRATS if a != b}
out["V4_best_of3_share"] = {s: round(float((cells.idxmax(axis=1) == s).mean()), 4) for s in STRATS}
out["V4_onlinelp_sum_is_largest"] = bool(v4.score_sum["OnlineLP"] > v4.score_sum["PID"] and v4.score_sum["OnlineLP"] > v4.score_sum["ABid"])
bc = ep.pivot_table(index="category", columns="strategy", values="score_component", aggfunc="sum")[STRATS]
be = ep.pivot_table(index="episode", columns="strategy", values="score_component", aggfunc="sum")[STRATS]
bc.to_csv(RES / "V4_by_category.csv")
be.to_csv(RES / "V4_by_episode.csv")
out["V4_best_by_category"] = {int(k): v for k, v in bc.idxmax(axis=1).items()}
out["V4_best_by_episode"] = {int(k): v for k, v in be.idxmax(axis=1).items()}
al = ticks.groupby(["strategy", "tick"]).alpha_eff.median().unstack(0)[STRATS]
al["others_median"] = ticks.groupby("tick").alpha_others_median.median()
nw = ticks.groupby(["strategy", "tick"]).n_won.median().unstack(0)[STRATS].add_prefix("n_won_median_")
al.join(nw).to_csv(RES / "V4_alpha_median_by_tick.csv")
out["V4_alpha_median"] = {s: dict(tick0=round(float(al[s].iloc[0]), 2), min=round(float(al[s].min()), 2), max=round(float(al[s].max()), 2),
                                  mean=round(float(al[s].mean()), 2),
                                  first_tick_median_eq0=int(al[s][al[s] == 0].index.min()) if (al[s] == 0).any() else -1,
                                  first_tick_n_won_median_gt0=int(nw["n_won_median_" + s][nw["n_won_median_" + s] > 0].index.min()))
                          for s in STRATS}
out["V4_alpha_others_median"] = dict(tick0=round(float(al.others_median.iloc[0]), 2), mean=round(float(al.others_median.mean()), 2),
                                     min=round(float(al.others_median.min()), 2), max=round(float(al.others_median.max()), 2))

# ---------------------------------------------------------------- V5 CPA 制約の達成状況
ep["cpa_ratio"] = ep.real_cpa / ep.cpa_constraint
g = ep.groupby("strategy")
v5 = pd.DataFrame(dict(
    n_cells=g.size(), exceed_share=g.penalty.apply(lambda x: (x < 1).mean()),
    exceedance_median=g.cpa_exceedance_rate.median(), n_exceedance_gt05=g.cpa_exceedance_rate.apply(lambda x: int((x > 0.5).sum())),
    lost_reward_share=1 - g.score_component.sum() / g.reward.sum(), cpa_ratio_median=g.cpa_ratio.median(),
    n_reward_zero=g.reward.apply(lambda x: int((x == 0).sum())), reward_median=g.reward.median())).loc[STRATS]
v5.to_csv(RES / "V5_cpa_by_strategy.csv")
out["V5"] = v5.round(4).to_dict("index")

# ---------------------------------------------------------------- V6 欠損
miss = {}
for name, df in (("episodes", ep.drop(columns="cpa_ratio")), ("ticks", ticks), ("agents", agents)):
    num = df.select_dtypes("number")
    miss[name] = {c: int((~np.isfinite(num[c])).sum()) for c in num.columns if (~np.isfinite(num[c])).any()}
out["V6_rows"] = dict(episodes=int(len(ep)), ticks=int(len(ticks)), agents=int(len(agents)))
out["V6_nonfinite_counts"] = miss
out["V6_ok"] = all(len(v) == 0 for v in miss.values())

# ---------------------------------------------------------------- 図
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
for s in STRATS:
    ax[0].plot(al.index, al[s], color=COLORS[s], label=LABEL[s], lw=1.4)
    t = ticks[ticks.strategy == s]
    ax[1].plot(t.groupby("tick").cost.sum().cumsum() / ep[ep.strategy == s].budget.sum(), color=COLORS[s], label=LABEL[s], lw=1.4)
    ax[2].plot(nw.index, nw["n_won_median_" + s], color=COLORS[s], label=LABEL[s], lw=1.4)
ax[0].plot(al.index, al.others_median, color="gray", ls=":", lw=1.2, label="他の47社の中央値")
ax[0].set_title("入札係数 α（192セルの中央値）", fontsize=10); ax[0].set_ylabel("α")
ax[1].set_title("予算の消化（累積の支払い ÷ 予算、192セルの合計）", fontsize=10); ax[1].set_ylabel("消化率")
ax[2].set_title("落札数（192セルの中央値）", fontsize=10); ax[2].set_ylabel("1ティックの落札数")
for a in ax:
    a.set_xlabel("ティック"); a.legend(fontsize=7.5); a.grid(alpha=0.3); a.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_alpha_time.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
ax[0].bar([LABEL[s] for s in STRATS], [v4.reward_mean[s] for s in STRATS], color=[COLORS[s] for s in STRATS], alpha=0.35, label="ペナルティ前の報酬")
ax[0].bar([LABEL[s] for s in STRATS], [v4.score_cell_mean[s] for s in STRATS], color=[COLORS[s] for s in STRATS], label="score")
ax[0].set_title("セルの平均（薄い色：ペナルティ前の報酬、濃い色：score）", fontsize=9.5); ax[0].set_ylabel("セルの平均")
parts = ax[1].boxplot([cells[s] for s in STRATS], labels=[LABEL[s] for s in STRATS], patch_artist=True, flierprops=dict(markersize=2))
for p, s in zip(parts["boxes"], STRATS):
    p.set_facecolor(COLORS[s]); p.set_alpha(0.6)
ax[1].set_title("セル（位置×エピソード）ごとの score", fontsize=10); ax[1].set_ylabel("score")
bins = np.linspace(-1, 3, 41)
for s in STRATS:
    ax[2].hist(ep[ep.strategy == s].cpa_exceedance_rate.clip(-1, 3), bins=bins, histtype="step", color=COLORS[s], lw=1.4, label=LABEL[s])
ax[2].axvline(0, color="k", lw=0.7)
ax[2].set_title("CPA の超過率（実績CPA−制約）÷制約。3以上は3に寄せた", fontsize=9.5)
ax[2].set_xlabel("超過率（正なら制約を超過）"); ax[2].set_ylabel("セルの数"); ax[2].legend(fontsize=7.5)
for a in ax:
    a.grid(alpha=0.3); a.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_scores.png", dpi=200); plt.close(fig)

(RES / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k.endswith("_ok") or k in ("use",)}, ensure_ascii=False))
