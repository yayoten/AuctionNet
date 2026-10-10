"""判定：今のコードの版で流した 18 run（rep=REP002new）が、Linux の既存の run と一致するか。

    .venv/bin/python research/W001_既存手法の比較/再実施_ルールベース/programs/compare_old_new.py

基準は REP.md の 3.2（実験の前に決めた）。episodes の全数値列の差が 0.0 なら一致。
"""
import json

import numpy as np
import pandas as pd

from common import HERE, OLD_VERSION, POS6, STRATS, default_rule_based, load_metas, read_table

RES = HERE / "results"
metas = default_rule_based(load_metas())
linux = metas[metas.os.str.startswith("Linux")]
old = linux[(linux.github_version == OLD_VERSION) & linux.player_index.isin(POS6)]
# 位置 0 の PID は、再現性の確認（repro_pid_p0）で --force の再実行をすると spec_name が変わるので、rep と位置で選ぶ
new = linux[(linux.rep == "REP002new") & linux.spec_name.isin(["check_same_as_old", "repro_pid_p0"]) & linux.player_index.isin(POS6)]
out = {"n_old": len(old), "n_new": len(new), "old_version": OLD_VERSION,
       "new_versions": sorted(new.github_version.unique()), "new_head_commits": sorted(new.head_commit.unique()),
       "new_status": new.status.value_counts().to_dict(), "new_dirty": int(new.github_dirty.sum()),
       "old_status": old.status.value_counts().to_dict(), "old_dirty": int(old.github_dirty.sum())}
key = ["strategy", "player_index"]
assert not old.duplicated(key).any() and not new.duplicated(key).any()


def table(m, name):
    t = read_table(m.run_id, name).merge(m[["run_id", "strategy", "player_index"]], on="run_id")
    return t.drop(columns="run_id")


eo, en = table(old, "episodes"), table(new, "episodes")
k = key + ["episode"]
j = eo.merge(en, on=k, suffixes=("_old", "_new"))
out["n_cells"] = len(j)
num = [c for c in eo.columns if c not in k and pd.api.types.is_numeric_dtype(eo[c])]
diff = {c: float((j[c + "_old"] - j[c + "_new"]).abs().max()) for c in num}
out["episodes_max_abs_diff"] = diff
bad = j[np.any([(j[c + "_old"] != j[c + "_new"]).values for c in num], axis=0)]
out["n_cells_mismatch"] = len(bad)
out["score_component_all_equal"] = bool((j.score_component_old == j.score_component_new).all())
cols = k + [c + s for c in ("score_component", "reward", "all_cost") for s in ("_old", "_new")]
j[cols].to_csv(RES / "B_cells_old_vs_new.csv", index=False)
bad[cols].to_csv(RES / "B_cells_mismatch.csv", index=False)

# ticks（判定には使わない。記録だけ）
to, tn = table(old, "ticks"), table(new, "ticks")
kt = k + ["tick"]
jt = to.merge(tn, on=kt, suffixes=("_old", "_new"))
skip = {"bidding_seconds", "tick_wall_seconds", "rss_mb"}
numt = [c for c in to.columns if c not in kt and c not in skip and pd.api.types.is_numeric_dtype(to[c])]
out["n_tick_rows"] = len(jt)
out["ticks_max_abs_diff"] = {c: float((jt[c + "_old"] - jt[c + "_new"]).abs().max()) for c in numt}
ao, an = table(old, "agents"), table(new, "agents")
ka = k + ["agent_index"]
ja = ao.merge(an, on=ka, suffixes=("_old", "_new"))
numa = [c for c in ao.columns if c not in ka and pd.api.types.is_numeric_dtype(ao[c])]
out["n_agent_rows"] = len(ja)
out["agents_max_abs_diff"] = {c: float((ja[c + "_old"].astype(float) - ja[c + "_new"].astype(float)).abs().max()) for c in numa}
out["judge"] = "一致" if (len(j) == 72 and len(bad) == 0 and (new.status == "ok").all() and not new.github_dirty.any()) else "不一致または測定不成立"
(RES / "B_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
