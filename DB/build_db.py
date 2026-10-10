"""DB/runs/ の記録から DuckDB（DB/auctionnet.duckdb）を作り直す。派生物なので、いつ消して作り直してもよい。

    .venv/bin/python DB/build_db.py   # Windows は .venv/Scripts/python.exe

原本は DB/runs/<run_id>/{params.json, meta.json, *.parquet}。ここで作る .duckdb は、それを SQL で横断して読むための索引。
表と列の意味は DB/columns.json（DB/README.md）。表・列には COMMENT として入れてある。
"""
import hashlib
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
# run ごとの parquet。表にするもの（小さい。git 管理）と、view にするもの（raw/ の大きいもの。git 管理外）
TABLE_FILES = {"episodes": "episodes.parquet", "ticks": "ticks.parquet", "agents": "agents.parquet",
               "sim_iters": "sim_iters.parquet"}
VIEW_FILES = {"agent_ticks": "raw/agent_ticks.parquet", "pv_won": "raw/pv_won.parquet", "pv_all": "raw/pv_all.parquet",
              "bids_all": "raw/bids_all_ep*.parquet"}


def config_key(spec):
    """設定だけのハッシュ（コードの版を含まない）。run_experiment.make_run_id と同じ項目・同じ並べ方。"""
    key = {k: spec.get(k) for k in ("player", "episodes", "seed", "gin")}
    return hashlib.sha1(json.dumps(key, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()[:10]


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
            m["config_key"] = config_key(p["spec"])
            flat = {}
            flatten("", {k: v for k, v in p["effective"].items() if k != "gin_bindings"}, flat)
            for k, v in flat.items():
                num = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
                longs.append(dict(run_id=m["run_id"], key=k, value_json=json.dumps(v, ensure_ascii=False), value_num=num))
        m.setdefault("record_version", 1)     # 版 1（2026-10-09 まで）の run には、この項目が無い
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
    q = lambda f: "'" + str(f).replace("\\", "/").replace("'", "''") + "'"
    for t, name in TABLE_FILES.items():
        files = [RUNS / r / name for r in ok if (RUNS / r / name).exists()]     # 版 1 の run には sim_iters が無い
        if files:
            con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet([{', '.join(q(f) for f in files)}], union_by_name=true)")
        else:
            print(f"警告: {t} に入れる run がありません")
    # raw/ の大きい記録は、コピーせず、parquet を直接読む view にする（無い run は入らない）
    for t, name in VIEW_FILES.items():
        files = [f for r in ok for f in sorted((RUNS / r).glob(name))]
        if files:
            con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet([{', '.join(q(f) for f in files)}], union_by_name=true)")
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
        # 同じ設定 × 位置で、いちばん新しい run（コードの版が違う run を、設定で結ぶ）
        con.execute("""
            CREATE VIEW runs_newest AS
            SELECT * EXCLUDE (rn) FROM (
              SELECT *, row_number() OVER (PARTITION BY config_key, player_index ORDER BY created_at DESC) AS rn
              FROM runs WHERE status = 'ok' AND github_dirty = false) WHERE rn = 1""")
        if "n_est" in [r[0] for r in con.execute("DESCRIBE agents").fetchall()]:
            # REP006 の三つの指標（恒等式 R = E × P × L）。全 48 社ぶん
            con.execute("""
                CREATE VIEW epl AS
                SELECT a.run_id, a.episode, a.agent_index, a.is_player, r.strategy, a.strategy_class, a.cpa_constraint,
                       a.cost, a.reward, a.n_est, a.n_real, a.real_var,
                       a.cost / nullif(a.n_est, 0) / a.cpa_constraint AS E,
                       a.n_est / nullif(a.n_real, 0) AS P,
                       a.n_real / nullif(a.reward, 0) AS L,
                       a.cost / nullif(a.reward, 0) / a.cpa_constraint AS R,
                       (a.reward - a.n_real) / sqrt(nullif(a.real_var, 0)) AS z
                FROM agents a JOIN runs r USING (run_id) WHERE a.n_est IS NOT NULL""")
    # 辞書を COMMENT に反映し、ずれを警告
    kinds = {r[0]: "VIEW" for r in con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()}
    for t, spec in cols.items():
        try:
            actual = [r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()]
        except Exception:
            continue
        kind = kinds.get(t, "TABLE")
        con.execute(f"COMMENT ON {kind} {t} IS '{spec['description'].replace(chr(39), chr(39)*2)}'")
        if not spec["columns"]:
            continue
        for c in actual:
            if c not in spec["columns"]:
                print(f"警告: {t}.{c} が columns.json にありません（DB/make_columns.py に書く）")
                continue
            s = spec["columns"][c]
            txt = f"{s['meaning']}（{s['unit']}）" if s["unit"] else s["meaning"]
            txt += f" ／ なぜ取るか: {s['why']}"
            try:
                con.execute(f"COMMENT ON COLUMN {t}.{c} IS '{txt.replace(chr(39), chr(39)*2)}'")
            except Exception:
                pass    # view の列に COMMENT を付けられない版の DuckDB では、columns.json を読む
        for c in spec["columns"]:
            if c not in actual and t not in ("runs", "models"):   # error / traceback などは、該当する行があるときだけ出来る
                print(f"警告: columns.json の {t}.{c} が実データにありません")
    n = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("runs", "episodes", "ticks", "agents", "sim_iters", "params_long", "models",
                                                                                 "agent_ticks", "pv_won")
         if t in [r[0] for r in con.execute("SHOW TABLES").fetchall()]}
    print("作成:", DB, n)
    con.close()


if __name__ == "__main__":
    sys.exit(main())
