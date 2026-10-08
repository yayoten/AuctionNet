"""REP005：REP002 の既定設定の run（8 手法 × 192 セル）を、CPA 制約の超過を主題にして数える。

    .venv/bin/python research/W001_.../REP005_.../programs/analyze.py

DB/auctionnet.duckdb を読むだけ（先に `python DB/build_db.py`）。新しい実験はしない。
集計の定義は REP005.md「2. 達成とみなす条件」に従う。ここでは、結果を見てから定義を変えない。
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
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
EPISODES = [0, 1, 2, 3]
RULE = ["PID", "ABid", "OnlineLP"]
LEARNED = ["BC", "IQL", "BCQ", "CQL", "TD3_BC"]
ORDER = RULE + LEARNED
CLASSES = ["超過", "制約内", "購入0・支払いあり", "購入0・支払いなし"]
CLASS_COLORS = ["#d62728", "#1f77b4", "#ff7f0e", "#c7c7c7"]
BIN_EDGES = [0, 0.1, 0.25, 0.5, 1.0, np.inf]          # 各区間は下端を含まず、上端を含む
BIN_LABELS = ["0〜0.1", "0.1〜0.25", "0.25〜0.5", "0.5〜1", "1超"]
QS = [0.10, 0.25, 0.50, 0.75, 0.90]

con = duckdb.connect(str(ROOT / "DB" / "auctionnet.duckdb"), read_only=True)
# REP002 の既定設定：ルールベースは既定のパラメータ、学習ベースは train_default（本家の既定のステップ数・seed 1）の自前の重み
runs = con.execute("""
    SELECT run_id, strategy, player_index, status, github_dirty, model_id, model_spec_name
    FROM runs
    WHERE rep = 'REP002'
      AND ((strategy IN ('PID', 'ABid', 'OnlineLP')) OR model_spec_name = 'train_default')
""").df()
ep = con.execute("SELECT e.* FROM episodes e JOIN runs r USING (run_id) WHERE r.rep = 'REP002'").df()
ep = ep.merge(runs[["run_id", "strategy", "player_index"]], on="run_id")
ep = ep[ep.episode.isin(EPISODES)].copy()
out = {}

# ---------------------------------------------------------------- S1 セルが揃っているか
out["S1_n_runs"] = runs.groupby("strategy").size().astype(int).to_dict()
out["S1_n_cells"] = ep.groupby("strategy").size().astype(int).to_dict()
out["S1_all_ok"] = bool((runs.status == "ok").all())
out["S1_any_dirty"] = bool(runs.github_dirty.any())
out["S1_no_duplicate_cells"] = bool(not ep.duplicated(["strategy", "player_index", "episode"]).any())
out["S1_pass"] = bool(all(out["S1_n_cells"].get(s) == 192 for s in ORDER) and out["S1_all_ok"]
                      and not out["S1_any_dirty"] and out["S1_no_duplicate_cells"])

# ---------------------------------------------------------------- S2 区分
has_conv = ep.reward > 0
ep["cpa"] = np.where(has_conv, ep.all_cost / ep.reward.where(has_conv), np.nan)
ep["x"] = ep.cpa / ep.cpa_constraint - 1          # 超過率（購入数が 0 なら NaN）
ep["cls"] = np.select(
    [has_conv & (ep.cpa > ep.cpa_constraint), has_conv, ep.all_cost > 0],
    CLASSES[:3], default=CLASSES[3])
cnt = ep.pivot_table(index="strategy", columns="cls", values="run_id", aggfunc="count", fill_value=0)
cnt = cnt.reindex(index=ORDER, columns=CLASSES, fill_value=0)
out["S2_pass"] = bool((cnt.sum(axis=1) == 192).all())
share = (cnt / 192).add_suffix("_割合")
by = pd.concat([cnt.add_suffix("_セル数"), share], axis=1)

# ---------------------------------------------------------------- S3 DB の超過率との一致
d = ep[has_conv]
diff = (d.cpa_exceedance_rate - d.x).abs()
out["S3_max_abs_diff"] = float(diff.max())
out["S3_pass"] = bool(diff.max() <= 1e-6)

# ---------------------------------------------------------------- 記録する量
# エピソード別の、超過したセルの割合（分母 48）
by_ep = (ep.assign(over=ep.cls == "超過").pivot_table(index="strategy", columns="episode", values="over", aggfunc="mean")
         .reindex(ORDER))
by_ep.columns = [f"超過の割合_ep{c}" for c in by_ep.columns]
by_ep.to_csv(RES / "S_over_share_by_episode.csv", encoding="utf-8")

# 超過率の分位点（購入数が 1 以上のセル）
q = d.groupby("strategy").x.quantile(QS).unstack().reindex(ORDER)
q.columns = [f"超過率_p{int(c * 100)}" for c in q.columns]
q["n_購入あり"] = d.groupby("strategy").size().reindex(ORDER).fillna(0).astype(int)
q.to_csv(RES / "S_exceedance_quantiles.csv", encoding="utf-8")

# 超過の大きさの区間別のセル数（超過したセルだけ）
over = ep[ep.cls == "超過"].copy()
over["bin"] = pd.cut(over.x, BIN_EDGES, labels=BIN_LABELS, right=True)
bins = over.pivot_table(index="strategy", columns="bin", values="run_id", aggfunc="count", fill_value=0, observed=False)
bins = bins.reindex(index=ORDER, columns=BIN_LABELS, fill_value=0).astype(int)
bins.to_csv(RES / "S_exceedance_bins.csv", encoding="utf-8")

# ペナルティ係数、ペナルティで失った報酬の割合、購入数
g = ep.groupby("strategy")
by["ペナルティ係数_平均"] = g.penalty.mean()
by["ペナルティ係数_超過セルの中央値"] = over.groupby("strategy").penalty.median()
by["ペナルティで失った報酬の割合"] = 1 - g.score_component.sum() / g.reward.sum().replace(0, np.nan)
by["reward_合計"] = g.reward.sum()
by["score_component_合計"] = g.score_component.sum()
for p in (10, 50, 90):
    by[f"購入数_p{p}"] = g.reward.quantile(p / 100)
by["DBでペナルティが掛かったセルの割合"] = g.penalty.apply(lambda s: (s < 1).mean())
by = by.reindex(ORDER)
by.to_csv(RES / "S_by_strategy.csv", encoding="utf-8")
ep[["strategy", "player_index", "episode", "cpa_constraint", "budget", "reward", "all_cost", "cpa", "x", "cls", "penalty",
    "score_component"]].sort_values(["strategy", "player_index", "episode"]).to_csv(RES / "S_cells.csv", index=False, encoding="utf-8")

out["by_strategy"] = by.round(4).where(by.notna(), None).to_dict("index")
out["over_share_by_episode"] = by_ep.round(4).to_dict("index")
out["exceedance_quantiles"] = q.round(4).where(q.notna(), None).to_dict("index")
out["exceedance_bins"] = bins.to_dict("index")
out["S4_pass"] = bool(set(by.index) == set(ORDER) and set(q.index) == set(ORDER) and set(bins.index) == set(ORDER))
out["all_pass"] = bool(out["S1_pass"] and out["S2_pass"] and out["S3_pass"] and out["S4_pass"])
(RES / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- 図
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
left = np.zeros(len(ORDER))
for c, col in zip(CLASSES, CLASS_COLORS):
    v = cnt[c].values / 192
    ax.barh(ORDER, v, left=left, color=col, label=c)
    left += v
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("セルの割合（192 セル）")
ax.set_title("(a) セルの区分")
ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)

ax = axes[1]
rng = np.random.default_rng(0)
CLIP = 2.0                                              # 図の表示だけ。これより大きい超過率は、右端に寄せて描く
for i, s in enumerate(ORDER):
    xs = d[d.strategy == s].x.values
    if len(xs) == 0:
        continue
    ys = i + rng.uniform(-0.3, 0.3, len(xs))
    xc = np.minimum(xs, CLIP)
    ax.scatter(xc, ys, s=5, alpha=0.5, color=np.where(xs > 0, CLASS_COLORS[0], CLASS_COLORS[1]), linewidths=0)
    if len(xs) >= 20:
        lo, med, hi = np.quantile(xs, [0.25, 0.5, 0.75])
        ax.plot([min(lo, CLIP), min(hi, CLIP)], [i, i], color="k", lw=2)
        ax.plot([min(med, CLIP)], [i], marker="|", color="k", ms=12, mew=2)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels(ORDER)
ax.invert_yaxis()
ax.set_xlim(-1, CLIP + 0.05)
ax.set_xlabel(f"超過率 = 実績 CPA ÷ 制約 − 1\n黒線は 25〜75% 点、縦棒は中央値。{CLIP:g} より大きい点は右端に描いた")
ax.set_title("(b) 超過率の分布（購入数が 1 以上のセル）")
fig.tight_layout()
fig.savefig(FIG / "fig_exceedance.png", dpi=200)

# 報告用（1 段の幅に収める）：超過率の分布だけ。落札した 6 手法
SIX = [s for s in ORDER if s not in ("CQL", "TD3_BC")]
fig, ax = plt.subplots(figsize=(5.2, 2.0))
rng = np.random.default_rng(0)
for i, s in enumerate(SIX):
    xs = d[d.strategy == s].x.values
    ys = i + rng.uniform(-0.3, 0.3, len(xs))
    ax.scatter(np.minimum(xs, CLIP), ys, s=5, alpha=0.5, color=np.where(xs > 0, CLASS_COLORS[0], CLASS_COLORS[1]), linewidths=0)
    lo, med, hi = np.quantile(xs, [0.25, 0.5, 0.75])
    ax.plot([lo, min(hi, CLIP)], [i, i], color="k", lw=2)
    ax.plot([med], [i], marker="|", color="k", ms=12, mew=2)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_yticks(range(len(SIX)))
ax.set_yticklabels(SIX)
ax.invert_yaxis()
ax.set_xlim(-1, CLIP + 0.05)
ax.set_xlabel("超過率 = 実績 CPA ÷ 制約 − 1")
fig.tight_layout()
fig.savefig(FIG / "fig_exceedance_rate.png", dpi=250)

print(json.dumps({k: out[k] for k in ("S1_pass", "S2_pass", "S3_pass", "S3_max_abs_diff", "S4_pass", "all_pass", "S1_n_cells")},
                 ensure_ascii=False))
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(by.round(3).T.to_string())
print(by_ep.round(3).to_string())
print(q.round(3).to_string())
print(bins.to_string())
