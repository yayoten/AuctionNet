"""プローブの測定（probe_policy.py の出力）から、学習の経過の表と図を作り、「落ち着いた」ステップ S* を決める。

    .venv/bin/python research/src/probe_policy.py --rep REP003 --out <REP003>/results/probe
    .venv/bin/python <REP003>/programs/analyze_probe.py <spec_name> [--tag loop1]

定義は REP003.md の 4 節のとおり（基準は α で 5。結果を見てから動かさない）。
- ステップ間の差：同じ seed で、ステップ S の α と、それ以降の各チェックポイントの α の、状態ごとの差の絶対値の平均（seed の平均）。
- シード間の差：同じステップで、seed の組ごとの、状態ごとの差の絶対値の平均（組の平均）。
- S* ：S 以降のすべてのチェックポイントで、ステップ間の差もシード間の差も基準以下になる、最小の S。
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RES, FIG = HERE / "results", HERE / "fig"
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
THRESHOLD = 5.0          # 「落ち着いた」の基準（α）。REP003.md 4 節
ORDER = ["BC", "IQL", "CQL", "BCQ", "TD3_BC"]
COLORS = {"BC": "#9467bd", "IQL": "#ff7f0e", "CQL": "#8c564b", "BCQ": "#e377c2", "TD3_BC": "#17becf"}

ap = argparse.ArgumentParser()
ap.add_argument("spec_names", nargs="+", help="対象にする学習の spec の name（複数可）")
ap.add_argument("--tag", default="loop1")
a = ap.parse_args()

idx = pd.read_csv(RES / "probe" / "probe_index.csv")
z = np.load(RES / "probe" / "probe_alpha.npz")
alpha, logged = z["alpha"], z["logged"]
idx["row"] = np.arange(len(idx))
idx = idx[idx.spec_name.isin(a.spec_names)]

rows, summary = [], {}
for algo in [x for x in ORDER if x in idx.algo.values]:
    g = idx[idx.algo == algo]
    steps = sorted(g.step_num.unique())
    seeds = sorted(g.seed.unique())
    A = {(r.seed, r.step_num): alpha[r.row] for r in g.itertuples()}
    for i, s in enumerate(steps):
        have = [sd for sd in seeds if (sd, s) in A]
        seed_diff = np.mean([np.abs(A[(x, s)] - A[(y, s)]).mean() for x, y in itertools.combinations(have, 2)]) if len(have) > 1 else np.nan
        later = [t for t in steps[i + 1:]]
        step_diff_max = max([np.mean([np.abs(A[(sd, s)] - A[(sd, t)]).mean() for sd in have if (sd, t) in A]) for t in later], default=0.0)
        step_diff_last = np.mean([np.abs(A[(sd, s)] - A[(sd, steps[-1])]).mean() for sd in have if (sd, steps[-1]) in A])
        sub = g[g.step_num == s]
        rows.append(dict(algo=algo, step_num=s, n_seeds=len(have), alpha_mean=sub.alpha_mean.mean(),
                         alpha_mean_min=sub.alpha_mean.min(), alpha_mean_max=sub.alpha_mean.max(),
                         alpha_std_over_states=sub.alpha_std.mean(), seed_diff=seed_diff,
                         step_diff_to_last=step_diff_last, step_diff_max_later=step_diff_max,
                         share_le0=sub.share_le0.mean(), share_le0_max=sub.share_le0.max(), share_ge99=sub.share_ge99.mean(),
                         corr_to_logged=sub.corr_to_logged.mean(), rmse_to_logged=sub.rmse_to_logged.mean()))
t = pd.DataFrame(rows)
# S*：S 以降のすべてのチェックポイントで、ステップ間の差もシード間の差も基準以下
for algo, g in t.groupby("algo"):
    g = g.sort_values("step_num")
    ok = (g.seed_diff <= THRESHOLD) & (g.step_diff_max_later <= THRESHOLD)
    tail_ok = ok[::-1].cummin()[::-1]                    # その行以降がすべて ok
    cand = g.step_num[tail_ok.values]
    last = g.iloc[-1]
    summary[algo] = dict(S_star=int(cand.min()) if len(cand) and len(g) > 1 and cand.min() < g.step_num.max() else None,
                         last_step=int(last.step_num), alpha_mean_last=round(float(last.alpha_mean), 2),
                         seed_diff_last=round(float(last.seed_diff), 2), share_ge99_last=round(float(last.share_ge99), 3),
                         share_le0_last=round(float(last.share_le0), 3), corr_to_logged_last=round(float(last.corr_to_logged), 3))
t["settled_from_here"] = False
for algo, v in summary.items():
    if v["S_star"] is not None:
        t.loc[(t.algo == algo) & (t.step_num >= v["S_star"]), "settled_from_here"] = True
t.to_csv(RES / f"{a.tag}_probe_by_step.csv", index=False)
out = dict(tag=a.tag, spec_names=a.spec_names, threshold_alpha=THRESHOLD, n_probe_states=int(alpha.shape[1]),
           logged_alpha_mean=round(float(logged.mean()), 2), logged_alpha_std=round(float(logged.std()), 2),
           logged_share_gt100=round(float((logged > 100).mean()), 3), by_algo=summary)
(RES / f"{a.tag}_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

# 図：左から、平均の α、最後の重みとの差、シード間の差、上限への集中
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
for algo, g in t.groupby("algo"):
    g = g.sort_values("step_num")
    c = COLORS[algo]
    ax[0].plot(g.step_num, g.alpha_mean, "o-", color=c, label=algo, ms=3)
    ax[0].fill_between(g.step_num, g.alpha_mean_min, g.alpha_mean_max, color=c, alpha=0.2)
    ax[1].plot(g.step_num[:-1], g.step_diff_to_last[:-1], "o-", color=c, ms=3)
    ax[2].plot(g.step_num, g.seed_diff, "o-", color=c, ms=3)
    ax[3].plot(g.step_num, g.share_ge99, "o-", color=c, ms=3)
ax[0].axhline(logged.mean(), color="k", ls=":", lw=1); ax[0].text(110, logged.mean() + 2, "ログの平均", fontsize=7)
ax[0].set_title("プローブへの α の平均（帯は seed の幅）")
ax[1].set_title("最後の重みとの差（状態ごとの |差| の平均）")
ax[2].set_title("シード間の差（状態ごとの |差| の平均）")
ax[3].set_title("|α| が 99 以上の状態の割合")
for k in (1, 2):
    ax[k].axhline(THRESHOLD, color="k", ls="--", lw=0.8); ax[k].set_yscale("symlog", linthresh=1)
for x in ax:
    x.set_xscale("log"); x.set_xlabel("学習ステップ")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / f"fig_{a.tag}_probe.png", dpi=200); plt.close(fig)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(t.round(2).to_string())
print(json.dumps(out, ensure_ascii=False, indent=1))
