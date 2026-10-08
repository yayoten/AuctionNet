"""DB/runs/ の記録から DuckDB（DB/auctionnet.duckdb）を作り直す。派生物なので、いつ消して作り直してもよい。

    .venv/bin/python DB/build_db.py   # Windows は .venv/Scripts/python.exe

原本は DB/runs/<run_id>/{params.json, meta.json, *.parquet}。ここで作る .duckdb は、それを SQL で横断して読むための索引。
表と列の意味は DB/columns.json（DB/README.md）。表・列には COMMENT として入れてある。
"""
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

H = Path(__file__).resolve().parent
RUNS = H / "runs"
MODELS = H / "models"
DB = H / "auctionnet.duckdb"
JSON_FIELDS = ("sweep_point", "repo_dirty_paths")


def flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out[prefix] = obj


def load_runs():
    rows, longs = [], []
    for d in sorted(RUNS.iterdir()):
        mp = d / "meta.json"
        if not d.is_dir() or d.name.endswith(".tmp") or not mp.exists():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        for f in JSON_FIELDS:
            m[f] = json.dumps(m.get(f), ensure_ascii=False)
        pp = d / "params.json"
        if pp.exists():
            p = json.loads(pp.read_text(encoding="utf-8"))
            m["params_spec"] = json.dumps(p["spec"], ensure_ascii=False)
            m["params_effective"] = json.dumps(p["effective"], ensure_ascii=False)
            m["spec_file"] = p["spec"].get("spec_file")
            flat = {}
            flatten("", {k: v for k, v in p["effective"].items() if k != "gin_bindings"}, flat)
            for k, v in flat.items():
                num = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
                longs.append(dict(run_id=m["run_id"], key=k, value_json=json.dumps(v, ensure_ascii=False), value_num=num))
        rows.append(m)
    return pd.DataFrame(rows), pd.DataFrame(longs, columns=["run_id", "key", "value_json", "value_num"])


def load_models():
    """DB/models/<model_id>/meta.json（学習した重みの記録）を 1 行ずつ。"""
    rows = []
    dirs = [d for d in sorted(MODELS.iterdir()) if d.is_dir() and not d.name.endswith(".tmp")] if MODELS.is_dir() else []
    dirs += [c for d in dirs for c in sorted((d / "ckpt").glob("*")) if c.is_dir()]     # 学習の途中の重み（チェックポイント）
    for d in dirs:
        mp = d / "meta.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        m.setdefault("is_checkpoint", False)
        if "checkpoints" in m:
            m["checkpoints"] = json.dumps(m["checkpoints"])
        for f in ("sweep_point", "repo_dirty_paths", "loss_last10pct_mean"):
            m[f] = json.dumps(m.get(f), ensure_ascii=False)
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    cols = json.loads((H / "columns.json").read_text(encoding="utf-8"))
    runs, longs = load_runs()
    if DB.exists():
        DB.unlink()
    con = duckdb.connect(str(DB))
    con.register("runs_df", runs)
    con.execute("CREATE TABLE runs AS SELECT * FROM runs_df")
    con.register("longs_df", longs)
    con.execute("CREATE TABLE params_long AS SELECT * FROM longs_df")
    models = load_models()
    if len(models):
        con.register("models_df", models)
        con.execute("CREATE TABLE models AS SELECT * FROM models_df")
    ok = runs[runs.status == "ok"].run_id.tolist() if len(runs) else []
    for t in ("episodes", "ticks", "agents"):
        files = [str(RUNS / r / f"{t}.parquet") for r in ok]
        if files:
            lst = ", ".join("'" + f.replace("\\", "/") + "'" for f in files)
            con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet([{lst}], union_by_name=true)")
        else:
            print(f"警告: {t} に入れる run がありません")
    # 便利な view：run 単位の要約
    if ok:
        con.execute("""
            CREATE VIEW run_summary AS
            SELECT r.run_id, r.spec_name, r.strategy, r.player_index, r.sweep_point, r.score, r.github_dirty,
                   e.n_ep, e.reward_mean, e.penalty_mean, e.real_cpa_mean, e.cpa_exceed_mean,
                   e.budget_consumer_ratio_mean, e.win_pv_ratio_mean, e.score_rank_mean,
                   r.sim_seconds, r.bidding_seconds_per_call_mean
            FROM runs r JOIN (
              SELECT run_id, count(*) n_ep, avg(reward) reward_mean, avg(penalty) penalty_mean, avg(real_cpa) real_cpa_mean,
                     avg(cpa_exceedance_rate) cpa_exceed_mean, avg(budget_consumer_ratio) budget_consumer_ratio_mean,
                     avg(win_pv_ratio) win_pv_ratio_mean, avg(score_rank) score_rank_mean
              FROM episodes GROUP BY run_id) e USING (run_id)""")
    # 辞書を COMMENT に反映し、ずれを警告
    for t, spec in cols.items():
        try:
            actual = [r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()]
        except Exception:
            continue
        con.execute(f"COMMENT ON TABLE {t} IS '{spec['description'].replace(chr(39), chr(39)*2)}'")
        for c in actual:
            if c not in spec["columns"]:
                print(f"警告: {t}.{c} が columns.json にありません（DB/make_columns.py に書く）")
                continue
            s = spec["columns"][c]
            txt = f"{s['meaning']}（{s['unit']}）" if s["unit"] else s["meaning"]
            txt += f" ／ なぜ取るか: {s['why']}"
            con.execute(f"COMMENT ON COLUMN {t}.{c} IS '{txt.replace(chr(39), chr(39)*2)}'")
        for c in spec["columns"]:
            if c not in actual and t not in ("runs", "models"):   # error / traceback などは、該当する行があるときだけ出来る
                print(f"警告: columns.json の {t}.{c} が実データにありません")
    n = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("runs", "episodes", "ticks", "agents", "params_long", "models")
         if t in [r[0] for r in con.execute("SHOW TABLES").fetchall()]}
    print("作成:", DB, n)
    con.close()


if __name__ == "__main__":
    sys.exit(main())
