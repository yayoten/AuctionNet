"""候補の設定を「試した」結果を集計し、6 節の合否の基準で判定する。

    .venv/bin/python <REP003>/programs/analyze_cand.py

DB/runs/ の原本（meta.json・params.json・parquet）を直接読む（rep = REP003、spec_name = cand_eval_*、status = ok）。
合格の条件（REP003.md 6 節。結果を見てから動かさない）：
 (i)  全手法・全 seed で落札している（予算消化率が 0 でない）。
 (ii) 手法ごとに、3 seed の「24 セルの平均 score」の最大と最小の差が、3 seed の平均の 10% 以内。
score は、セル（位置 × エピソード）ごとの score_component（ペナルティ係数 × 獲得価値）。
出力：results/cand_by_seed.csv、results/cand_summary.json
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[2]
RES = HERE / "results"
ORDER = ["BC", "IQL", "CQL", "BCQ", "TD3_BC"]
LIMIT = 0.10

rows = []
for mp in sorted((REPO / "DB" / "runs").glob("R*/meta.json")):
    m = json.loads(mp.read_text(encoding="utf-8"))
    if m.get("rep") != "REP003" or not str(m.get("spec_name", "")).startswith("cand_eval_") or m.get("status") != "ok":
        continue
    spec = json.loads((mp.parent / "params.json").read_text(encoding="utf-8"))["spec"]
    md = spec["player"]["kwargs"]["model_dir"]
    mm = json.loads((REPO / md / "meta.json").read_text(encoding="utf-8"))
    ep = pd.read_parquet(mp.parent / "episodes.parquet")
    tk = pd.read_parquet(mp.parent / "ticks.parquet")
    al = tk.groupby("episode").apply(lambda g: pd.Series(dict(alpha_eff_mean=g.alpha_eff.mean(), alpha_eff_std_over_ticks=g.alpha_eff.std(),
                                                             alpha_eff_min=g.alpha_eff.min(), alpha_eff_max=g.alpha_eff.max())))
    ep = ep.merge(al, left_on="episode", right_index=True)
    ep = ep.assign(algo=mm["algo"], model_seed=mm["seed"], model_id=mm["model_id"], model_step_num=mm["step_num"],
                   player_index=int(m["player_index"]), github_dirty=str(m.get("github_dirty")), head_commit=m.get("head_commit"))
    rows.append(ep)
cells = pd.concat(rows, ignore_index=True)
cells.to_csv(RES / "cand_cells.csv", index=False)
by = (cells.groupby(["algo", "model_seed", "model_id"])
      .agg(n_cells=("score_component", "size"), score_mean=("score_component", "mean"), reward_mean=("reward", "mean"),
           penalty_mean=("penalty", "mean"), budget_ratio_mean=("budget_consumer_ratio", "mean"),
           budget_ratio_min=("budget_consumer_ratio", "min"), cpa_exceed_share=("cpa_exceedance_rate", lambda x: float((x > 0).mean())),
           alpha_eff_mean=("alpha_eff_mean", "mean"), alpha_eff_std_over_ticks=("alpha_eff_std_over_ticks", "mean"),
           alpha_eff_min=("alpha_eff_min", "min"), alpha_eff_max=("alpha_eff_max", "max"),
           dirty=("github_dirty", lambda x: ",".join(sorted(set(x)))))
      .reset_index())
by["algo"] = pd.Categorical(by.algo, [a for a in ORDER if a in set(by.algo)])
by = by.sort_values(["algo", "model_seed"])
by.to_csv(RES / "cand_by_seed.csv", index=False)
summary = {}
for algo, g in by.groupby("algo", observed=True):
    mean = float(g.score_mean.mean())
    spread = float((g.score_mean.max() - g.score_mean.min()) / mean) if mean else float("nan")
    summary[str(algo)] = dict(n_seeds=int(len(g)), n_cells_per_seed=[int(x) for x in g.n_cells], score_mean_by_seed=[round(float(x), 3) for x in g.score_mean],
                              score_mean=round(mean, 3), spread=round(spread, 4),
                              pass_i_all_seeds_win=bool((g.budget_ratio_min > 0).all()), pass_ii_spread_within_10pct=bool(spread <= LIMIT),
                              complete=bool(len(g) == 3 and (g.n_cells == 24).all()),
                              alpha_eff_mean_by_seed=[round(float(x), 2) for x in g.alpha_eff_mean])
(RES / "cand_summary.json").write_text(json.dumps(dict(limit=LIMIT, by_algo=summary), ensure_ascii=False, indent=1), encoding="utf-8")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(by.round(3).to_string())
print(json.dumps(summary, ensure_ascii=False, indent=1))
