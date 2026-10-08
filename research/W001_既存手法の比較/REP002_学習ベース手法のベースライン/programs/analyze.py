"""REP002 の結果を DB から集計して、results/ に表（CSV, summary.json）、fig/ に図を作る。

    .venv/bin/python research/W001_.../REP002_.../programs/analyze.py

DB/auctionnet.duckdb を読む（先に `python DB/build_db.py`）。
集計の定義は REP002.md「2. 達成とみなす条件」に従う。ここでは、結果を見てから定義を変えない。
使うのは、この REP で流した run（runs.rep = 'REP002'。すべて同じ端末）だけ。
"""
import json
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
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "Yu Gothic", "Meiryo", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
EPISODES = [0, 1, 2, 3]
POS6 = [0, 8, 16, 24, 32, 40]
RULE = ["ABid", "PID", "OnlineLP"]
LEARNED = ["BC", "IQL", "CQL", "BCQ", "TD3_BC"]
ORDER = RULE + LEARNED
COLORS = {"PID": "#1f77b4", "ABid": "#2ca02c", "OnlineLP": "#d62728", "BC": "#9467bd", "IQL": "#ff7f0e",
          "CQL": "#8c564b", "BCQ": "#e377c2", "TD3_BC": "#17becf"}
# 論文 Figure 10 の目視の概算（REP001.md 10.）。ABid を 1.0 とした相対値。（基本タスク, 目標CPAタスク）
PAPER = {"OnlineLP": (7.0, 6.5), "IQL": (3.0, 2.6), "PID": (3.5, 2.3), "BC": (2.5, 2.3), "ABid": (1.0, 0.4)}

con = duckdb.connect(str(ROOT / "DB" / "auctionnet.duckdb"), read_only=True)
runs = con.execute("SELECT * FROM runs WHERE rep = 'REP002'").df()
models = con.execute("SELECT * FROM models WHERE rep = 'REP002'").df()
ep = con.execute("SELECT e.* FROM episodes e JOIN runs r USING (run_id) WHERE r.rep = 'REP002'").df()
ticks = con.execute("SELECT t.* FROM ticks t JOIN runs r USING (run_id) WHERE r.rep = 'REP002'").df()
agents = con.execute("SELECT a.* FROM agents a JOIN runs r USING (run_id) WHERE r.rep = 'REP002'").df()
out = {}


def variant(r):
    """run を、L4・L5 の条件に振り分ける。"""
    if r.strategy in RULE:
        return "default"                       # ルールベースの既定設定
    if pd.isna(r.model_id):
        return "bundled"                       # 同梱の重み（追加 1）
    return {"train_default": "default", "train_steps": "steps20000", "train_seeds": "seed"}[r.model_spec_name]


runs["variant"] = runs.apply(variant, axis=1)
runs = runs.merge(models[["model_id", "seed", "step_num"]].rename(columns={"seed": "model_seed"}), on="model_id", how="left")
runs.loc[runs.variant == "seed", "variant"] = "seed" + runs.model_seed[runs.variant == "seed"].astype(int).astype(str)
ok = runs[runs.status == "ok"]

# ---------------------------------------------------------------- L1 データ（DB/train_data の要約）
rows = []
for p in sorted((ROOT / "DB" / "train_data").glob("period-*.json"), key=lambda p: int(p.stem.split("-")[1])):
    s = json.loads(p.read_text(encoding="utf-8"))
    adv = pd.DataFrame(s["advertisers"])
    rows.append(dict(period=s["period"], n_rows=s["n_rows"], n_pv=s["n_pv"], n_nan=s["n_nan"], rl_rows=s["rl_rows"],
                     pvalue_mean=s["pvalue_mean"], lwc_mean=s["least_winning_cost_mean"],
                     alpha_mean_median=adv.alpha_mean.median(), alpha_tick0_median=adv.alpha_tick0.median(),
                     cpa_exceed_share=float((adv.real_cpa > adv.cpa_constraint).mean()),
                     budget_ratio_mean=float((adv.cost / adv.budget).mean())))
l1 = pd.DataFrame(rows)
l1.to_csv(RES / "L1_train_data_by_period.csv", index=False, encoding="utf-8")
allinfo = json.loads((ROOT / "DB" / "train_data" / "training_data_all.json").read_text(encoding="utf-8"))
out["L1"] = dict(n_periods=int(len(l1)), n_rows_total=int(l1.n_rows.sum()), n_nan_total=int(l1.n_nan.sum()),
                 n_pv_min=int(l1.n_pv.min()), n_pv_max=int(l1.n_pv.max()), rl_rows=int(allinfo["rows"]),
                 rl_sha1=allinfo["sha1"], alpha_mean_median=float(l1.alpha_mean_median.median()),
                 cpa_exceed_share_mean=float(l1.cpa_exceed_share.mean()), budget_ratio_mean=float(l1.budget_ratio_mean.mean()))

# ---------------------------------------------------------------- L2 学習
m = models.copy()
m["variant"] = m.spec_name.map({"train_default": "default", "train_steps": "steps20000", "train_seeds": "seed"})
m.loc[m.variant == "seed", "variant"] = "seed" + m.seed[m.variant == "seed"].astype(str)
m[["model_id", "algo", "variant", "step_num", "seed", "status", "train_seconds", "loss_all_finite", "loss_last10pct_mean",
   "model_sha1", "github_dirty", "device"]].sort_values(["algo", "variant"]).to_csv(RES / "L2_models.csv", index=False, encoding="utf-8")
out["L2"] = dict(n_models=int(len(m)), n_error=int((m.status != "ok").sum()), all_loss_finite=bool(m.loss_all_finite.all()),
                 any_dirty=bool(m.github_dirty.any()),
                 train_seconds={f"{r.algo}/{r.variant}": round(float(r.train_seconds), 1) for r in m.itertuples()})
out["L2_seconds_per_step_default"] = {r.algo: round(float(r.train_seconds) / r.step_num, 4)
                                      for r in m[m.variant == "default"].itertuples()}

# ---------------------------------------------------------------- L3 動くか
out["L3_runs_by_strategy_variant_status"] = {f"{k[0]}/{k[1]}/{k[2]}": int(v) for k, v in
                                             runs.groupby(["strategy", "variant", "status"]).size().items()}
out["L3_n_runs"] = int(len(runs))
out["L3_n_error"] = int((runs.status != "ok").sum())
out["L3_all_scores_finite"] = bool(np.isfinite(ok.score).all())
if (runs.status != "ok").any():
    runs[runs.status != "ok"][["run_id", "strategy", "variant", "player_index", "error"]].to_csv(RES / "L3_errors.csv", index=False,
                                                                                               encoding="utf-8")

# ---------------------------------------------------------------- 規模感
sc = ok.groupby("strategy").agg(n_runs=("run_id", "count"), sim_seconds_mean=("sim_seconds", "mean"),
                                sim_seconds_max=("sim_seconds", "max"), peak_rss_mb_max=("peak_rss_mb", "max"),
                                bidding_ms_per_call_mean=("bidding_seconds_per_call_mean", lambda x: 1000 * x.mean()))
sc["bidding_ms_max_single_call"] = ticks.merge(runs[["run_id", "strategy"]], on="run_id").groupby("strategy").bidding_seconds.max() * 1000
sc.to_csv(RES / "scale.csv", encoding="utf-8")
out["scale"] = sc.round(3).to_dict("index")

# ---------------------------------------------------------------- L4 8手法の比較（既定設定、48位置×4エピソード）
base = ok[ok.variant == "default"]
base_ep = ep.merge(base[["run_id", "strategy", "player_index"]], on="run_id")
cells = base_ep.pivot_table(index=["player_index", "episode"], columns="strategy", values="score_component")
out["L4_n_cells_per_strategy"] = cells.notna().sum().astype(int).to_dict()
cells = cells.dropna()
cells = cells[[s for s in ORDER if s in cells.columns]]
out["L4_n_cells"] = int(len(cells))
tot = cells.sum()
out["L4_total_score_component"] = tot.round(4).to_dict()
out["L4_total_score_normalized_20000"] = (tot / 20000).round(6).to_dict()
out["L4_relative_to_ABid"] = (tot / tot["ABid"]).round(4).to_dict() if "ABid" in tot else None
out["L4_rank_by_total"] = tot.sort_values(ascending=False).index.tolist()
out["L4_paper_fig10_relative_to_ABid_basic_and_cpa"] = PAPER
cells.to_csv(RES / "L4_cells_score_component.csv", encoding="utf-8")
agg = base_ep.groupby("strategy").agg(
    score_component_sum=("score_component", "sum"), score_component_mean=("score_component", "mean"),
    reward_mean=("reward", "mean"), penalty_mean=("penalty", "mean"), penalty_lt1_share=("penalty", lambda x: (x < 1).mean()),
    real_cpa_median=("real_cpa", "median"), cpa_exceedance_median=("cpa_exceedance_rate", "median"),
    budget_ratio_mean=("budget_consumer_ratio", "mean"), budget_lt09_share=("budget_consumer_ratio", lambda x: (x < 0.9).mean()),
    stopped_early_share=("last_compete_tick_index", lambda x: (x < 47).mean()),
    win_pv_ratio_mean=("win_pv_ratio", "mean"), score_rank_median=("score_rank", "median"),
    top10_share=("score_rank", lambda x: (x <= 10).mean()), others_score_mean=("others_score_mean", "mean"))
agg["relative_to_ABid"] = agg.score_component_sum / agg.loc["ABid", "score_component_sum"] if "ABid" in agg.index else np.nan
agg["cell_cv"] = cells.std() / cells.mean()
agg = agg.reindex([s for s in ORDER if s in agg.index])
agg.to_csv(RES / "L4_by_strategy.csv", encoding="utf-8")
out["L4_by_strategy"] = agg.round(4).to_dict("index")
if "OnlineLP" in cells.columns:
    out["L4_cell_win_rate_vs_OnlineLP"] = {s: float((cells[s] > cells.OnlineLP).mean()) for s in cells.columns if s != "OnlineLP"}
if len(cells):
    out["L4_cell_best_share"] = cells.idxmax(axis=1).value_counts(normalize=True).round(4).to_dict()
base_ep["category"] = base_ep["category"].astype(int)
by_cat = base_ep.groupby(["category", "strategy"]).score_component.sum().unstack()
by_cat.to_csv(RES / "L4_by_category.csv", encoding="utf-8")
out["L4_best_strategy_by_category"] = {int(k): v for k, v in by_cat.idxmax(axis=1).items()}
by_ep = base_ep.groupby(["episode", "strategy"]).score_component.sum().unstack()
by_ep.to_csv(RES / "L4_by_episode.csv", encoding="utf-8")
out["L4_best_strategy_by_episode"] = {int(k): v for k, v in by_ep.idxmax(axis=1).items()}
out["L4_learned_best_by_category"] = {int(k): v for k, v in by_cat[[s for s in LEARNED if s in by_cat]].idxmax(axis=1).items()}
# α の水準（ティックごとの中央値の、時間方向の要約）
tb = ticks.merge(base[["run_id", "strategy"]], on="run_id")
al = tb.groupby(["strategy", "tick"]).alpha_eff.median().unstack(0)
al.to_csv(RES / "L4_alpha_median_by_tick.csv", encoding="utf-8")
out["L4_alpha_median"] = {s: dict(tick0=round(float(al[s].iloc[0]), 2), min=round(float(al[s].min()), 2),
                                  max=round(float(al[s].max()), 2), mean=round(float(al[s].mean()), 2),
                                  last_tick_gt0=int(al[s][al[s] > 0].index.max()) if (al[s] > 0).any() else -1)
                          for s in al.columns}
out["L4_alpha_others_median_mean"] = round(float(tb.groupby("tick").alpha_others_median.median().mean()), 2)

# ---------------------------------------------------------------- L5 学習の条件への感度（6位置×4エピソード）
six = ok[ok.player_index.isin(POS6)].merge(ep, on="run_id")
l5 = six.groupby(["strategy", "variant"]).agg(n_cells=("score_component", "size"), score_component_mean=("score_component", "mean"),
                                              reward_mean=("reward", "mean"), penalty_mean=("penalty", "mean"),
                                              penalty_lt1_share=("penalty", lambda x: (x < 1).mean()),
                                              budget_ratio_mean=("budget_consumer_ratio", "mean")).reset_index()
d = l5[l5.variant == "default"].set_index("strategy").score_component_mean
l5["ratio_to_default"] = l5.score_component_mean / l5.strategy.map(d)
l5 = l5.set_index("strategy").loc[[s for s in ORDER if s in l5.strategy.values]].reset_index()
l5.to_csv(RES / "L5_variants.csv", index=False, encoding="utf-8")
out["L5"] = {f"{r.strategy}/{r.variant}": dict(n_cells=int(r.n_cells), score_component_mean=round(r.score_component_mean, 3),
                                               ratio_to_default=round(r.ratio_to_default, 3)) for r in l5.itertuples()}
# 手法間の差と、シードによる差の比較に使う量（定義だけ記録する。判断は付けない）
seedv = l5[l5.variant.isin(["default", "seed2", "seed3"]) & l5.strategy.isin(LEARNED)]
out["L5_seed_range_ratio"] = {s: [round(float(g.ratio_to_default.min()), 3), round(float(g.ratio_to_default.max()), 3)]
                              for s, g in seedv.groupby("strategy")}

# ---------------------------------------------------------------- L6 欠損
miss = {}
for name, df in (("episodes", ep), ("ticks", ticks), ("agents", agents)):
    num = df.select_dtypes("number")
    miss[name] = {c: int((~np.isfinite(num[c])).sum()) for c in num.columns if (~np.isfinite(num[c])).any()}
out["L6_rows"] = dict(episodes=int(len(ep)), ticks=int(len(ticks)), agents=int(len(agents)))
out["L6_nonfinite_counts"] = miss
out["L6_ok"] = all(len(v) == 0 for v in miss.values())


# ---------------------------------------------------------------- 図
def fig_scores():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    order = list(cells.columns)
    ax[0].bar(order, [cells[s].sum() / 20000 for s in order], color=[COLORS[s] for s in order])
    ax[0].set_ylabel("score（20000で割った合計）")
    ax[0].set_title("48位置×4エピソードの合計")
    ax[0].axvline(2.5, color="gray", lw=0.7, ls=":")
    parts = ax[1].boxplot([cells[s] for s in order], tick_labels=order, showfliers=True, patch_artist=True, flierprops=dict(markersize=2))
    for p, s in zip(parts["boxes"], order):
        p.set_facecolor(COLORS[s]); p.set_alpha(0.6)
    ax[1].set_yscale("symlog", linthresh=1)
    ax[1].set_ylabel("セルごとの score"); ax[1].set_title("セル（位置×エピソード）のばらつき")
    ax[1].axvline(3.5, color="gray", lw=0.7, ls=":")
    for a in ax:
        a.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_scores.png", dpi=200); plt.close(fig)


def fig_alpha():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for s in [x for x in ORDER if x in tb.strategy.unique()]:
        g = tb[tb.strategy == s].groupby("tick")
        ls = "-" if s in LEARNED else "--"
        ax[0].plot(g.alpha_eff.median(), ls, color=COLORS[s], label=s, lw=1.3)
        ax[1].plot(g.cost.sum().cumsum() / (tb[tb.strategy == s].merge(base_ep[["run_id", "episode", "budget"]].drop_duplicates(),
                                                                         on=["run_id", "episode"]).drop_duplicates(["run_id", "episode"]).budget.sum()),
                   ls, color=COLORS[s], label=s, lw=1.3)
        ax[2].plot(g.n_won.mean(), ls, color=COLORS[s], label=s, lw=1.3)
    ax[0].plot(tb.groupby("tick").alpha_others_median.median(), "k:", label="他47社の中央値", lw=1.5)
    ax[0].set_yscale("symlog", linthresh=10); ax[0].set_title("実効の入札係数 α（中央値）"); ax[0].set_xlabel("ティック")
    ax[1].set_title("予算に対する支払いの累計"); ax[1].set_xlabel("ティック")
    ax[2].set_title("落札数（平均）"); ax[2].set_xlabel("ティック"); ax[2].set_yscale("symlog", linthresh=1)
    ax[2].legend(fontsize=6.5, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "fig_alpha_time.png", dpi=200); plt.close(fig)


def fig_variants():
    v = l5[l5.strategy.isin(LEARNED)]
    names = ["default", "seed2", "seed3", "steps20000", "bundled"]
    labels = ["既定(seed1)", "seed2", "seed3", "20000ステップ", "同梱の重み"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.4))
    w = 0.16
    algos = [s for s in LEARNED if s in v.strategy.values]
    for i, (n, lab) in enumerate(zip(names, labels)):
        y = [v[(v.strategy == s) & (v.variant == n)].score_component_mean.sum() if ((v.strategy == s) & (v.variant == n)).any() else np.nan
             for s in algos]
        ax.bar(np.arange(len(algos)) + (i - 2) * w, y, w, label=lab)
    for s, c in (("OnlineLP", COLORS["OnlineLP"]), ("ABid", COLORS["ABid"]), ("PID", COLORS["PID"])):
        r = l5[(l5.strategy == s) & (l5.variant == "default")]
        if len(r):
            ax.axhline(r.score_component_mean.iloc[0], color=c, lw=0.9, ls="--")
            ax.text(len(algos) - 0.45, r.score_component_mean.iloc[0], s, color=c, fontsize=7, va="bottom")
    ax.set_xticks(np.arange(len(algos))); ax.set_xticklabels(algos)
    ax.set_ylabel("セルの平均 score（6位置×4エピソード）"); ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(); fig.savefig(FIG / "fig_variants.png", dpi=200); plt.close(fig)


def fig_loss():
    ms = m[m.variant.isin(["default", "steps20000"])].sort_values(["algo", "variant"])
    algos = [s for s in LEARNED if s in ms.algo.values]
    fig, axes = plt.subplots(1, len(algos), figsize=(2.6 * len(algos), 2.8))
    for ax, s in zip(np.atleast_1d(axes), algos):
        for r in ms[ms.algo == s].itertuples():
            loss = pd.read_csv(ROOT / "DB" / "models" / r.model_id / "loss.csv")
            col = "Action_loss" if "Action_loss" in loss else "Q_loss"
            y = loss[col].rolling(max(1, len(loss) // 100), min_periods=1).mean()
            ax.plot(loss.step + 1, y, lw=1, label=f"{r.step_num}ステップ")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"{s}（{'行動の誤差' if s == 'BC' else 'Q の損失'}）", fontsize=8)
        ax.set_xlabel("学習ステップ", fontsize=7); ax.tick_params(labelsize=6); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "fig_loss.png", dpi=200); plt.close(fig)


if len(cells):
    fig_scores(); fig_alpha()
if len(l5):
    fig_variants()
if len(m):
    fig_loss()
(RES / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
