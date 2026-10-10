"""DB/runs の原本（meta.json と parquet）を直接読む。DB/auctionnet.duckdb（索引）は使わない。

索引は作り直しが要り、並行して動く別のタスクと衝突しうるため。
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parents[2]
RUNS = ROOT / "DB" / "runs"
STRATS = ["PID", "ABid", "OnlineLP"]
EPISODES = [0, 1, 2, 3]
POS6 = [0, 8, 16, 24, 32, 40]
OLD_VERSION = "0536c690ef17edb6931655160fe3033bc2aac44e"   # Linux の既存の 144 run の github の版


def load_metas():
    """全 run の meta と、spec（player の kwargs・episodes・gin）を 1 行ずつ。"""
    rows = []
    for mp in sorted(RUNS.glob("*/meta.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        pp = mp.parent / "params.json"
        spec = json.loads(pp.read_text(encoding="utf-8"))["spec"] if pp.exists() else {}
        m["kwargs"] = json.dumps(spec.get("player", {}).get("kwargs", {}), sort_keys=True)
        m["episodes_list"] = json.dumps(spec.get("episodes"))
        m["gin"] = json.dumps(spec.get("gin") or {})
        m["seed"] = spec.get("seed")
        m.pop("traceback", None)
        rows.append(m)
    return pd.DataFrame(rows)


def default_rule_based(metas):
    """既定設定（kwargs が空・gin の上書きなし・エピソード 0〜3・seed 1）のルールベースの run。"""
    return metas[metas.strategy.isin(STRATS) & (metas.kwargs == "{}") & (metas.gin == "{}")
                 & (metas.episodes_list == json.dumps(EPISODES)) & (metas.seed == 1)]


def read_table(run_ids, name):
    return pd.concat([pd.read_parquet(RUNS / r / f"{name}.parquet") for r in run_ids], ignore_index=True)
