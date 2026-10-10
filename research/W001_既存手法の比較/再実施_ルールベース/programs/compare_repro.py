"""再現性（V3）：PID・位置 0 を --force で再実行した結果を、再実行の前の写し（results/raw/repro_before/）と比べる。

    .venv/bin/python research/W001_既存手法の比較/再実施_ルールベース/programs/compare_repro.py
"""
import json

import pandas as pd

from common import HERE, RUNS

RUN = "Rbce18fda09-p00"
before, after = HERE / "results" / "raw" / "repro_before", RUNS / RUN
mb, ma = (json.loads((d / "meta.json").read_text(encoding="utf-8")) for d in (before, after))
out = {"run_id": RUN, "before": {k: mb[k] for k in ("created_at", "spec_name", "github_version", "status", "sim_seconds")},
       "after": {k: ma[k] for k in ("created_at", "spec_name", "github_version", "status", "sim_seconds")}}
assert mb["created_at"] != ma["created_at"], "再実行されていない"
skip = {"bidding_seconds", "tick_wall_seconds", "rss_mb"}
for name in ("episodes", "ticks", "agents"):
    b, a = pd.read_parquet(before / f"{name}.parquet"), pd.read_parquet(after / f"{name}.parquet")
    num = [c for c in b.columns if c not in skip and pd.api.types.is_numeric_dtype(b[c])]
    other = [c for c in b.columns if c not in skip and c not in num]
    out[name] = {"rows": [len(b), len(a)], "max_abs_diff": {c: float((b[c].astype(float) - a[c].astype(float)).abs().max()) for c in num},
                 "all_equal": bool(len(b) == len(a) and all((b[c] == a[c]).all() for c in num + other))}
out["V3_ok"] = bool(out["episodes"]["all_equal"] and mb["status"] == ma["status"] == "ok")
(HERE / "results" / "V3_repro.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
