"""04: 版 1（2026-10-09 まで）の run と、版 2（T006）の run が、同じ DuckDB に入ること。

- 版 1 の run は、DB/runs/ の実物を tmp に写して使う（原本は読むだけ）。
- 既存の表（runs / episodes / ticks / agents）の既存の列は、そのまま残る。版 1 の run の新しい列は NULL。
- 新しい表・view（sim_iters / agent_ticks / pv_won / pv_all / epl / runs_newest）ができ、列の辞書（columns.json）とずれが無い。
"""
import json
import shutil

import duckdb
import pandas as pd
import pytest

import build_db
import make_columns
from conftest import DB_DIR, run_one

pytestmark = pytest.mark.slow
OLD_COLS = {"episodes": ["reward", "all_cost", "real_cpa", "penalty", "score_component", "cpa_exceedance_rate"],
            "ticks": ["alpha_eff", "n_won", "n_exposed", "reward", "cost", "remaining_budget_before", "bidding_seconds"],
            "agents": ["agent_name", "reward", "cost", "real_cpa", "penalty", "score_component", "is_player"]}


def old_run_dirs(n=2):
    """DB/runs/ にある、版 1 の run（record_version が無く、status = ok）を n 個。"""
    out = []
    for d in sorted((DB_DIR / "runs").iterdir()) if (DB_DIR / "runs").is_dir() else []:
        mp = d / "meta.json"
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            if m.get("status") == "ok" and "record_version" not in m and (d / "ticks.parquet").exists():
                out.append(d)
                if len(out) == n:
                    break
    return out


@pytest.fixture(scope="module")
def db(tmp_path_factory, full_runs):
    root = tmp_path_factory.mktemp("db")
    runs = root / "runs"
    runs.mkdir()
    old = old_run_dirs()
    for d in old:
        shutil.copytree(d, runs / d.name)
    for s in ("PID", "IQL"):
        shutil.copytree(full_runs[s][1], runs / full_runs[s][1].name)
    m_basic, d_basic = run_one(root / "b", "ABid", "basic", episodes=(0,))
    shutil.copytree(d_basic, runs / d_basic.name)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(build_db, "RUNS", runs)
        mp.setattr(build_db, "MODELS", root / "no_models")
        mp.setattr(build_db, "DB", root / "t.duckdb")
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            build_db.main()
    con = duckdb.connect(str(root / "t.duckdb"), read_only=True)
    yield dict(con=con, old=[d.name for d in old], new=[full_runs["PID"][0]["run_id"], full_runs["IQL"][0]["run_id"]],
               basic=m_basic["run_id"], log=buf.getvalue())
    con.close()


def test_columns_json_is_up_to_date():
    saved = json.loads((DB_DIR / "columns.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(make_columns.COLUMNS, ensure_ascii=False))     # make_columns.py を直したら、作り直す
    for t in ("sim_iters", "agent_ticks", "pv_won", "pv_all", "epl", "runs_newest"):
        assert t in saved
    for t, spec in saved.items():
        for name, col in spec["columns"].items():
            assert col["meaning"] and col["why"], (t, name)                              # 意味と「なぜ取るか」が、すべての列にある


def test_no_column_is_missing_from_the_dictionary(db):
    assert "が columns.json にありません" not in db["log"], db["log"]


def test_old_and_new_runs_are_in_the_same_tables(db):
    con = db["con"]
    runs = con.execute("SELECT run_id, record_version, record_level, config_key FROM runs").df().set_index("run_id")
    assert len(runs) == len(db["old"]) + 3 and runs.config_key.notna().all()
    for r in db["old"]:
        assert runs.record_version[r] == 1 and pd.isna(runs.record_level[r])
    for r in db["new"]:
        assert runs.record_version[r] == 2 and runs.record_level[r] == "full"
    for t in ("episodes", "ticks", "agents"):
        got = set(con.execute(f"SELECT DISTINCT run_id FROM {t}").df().run_id)
        assert got == set(runs.index), t


@pytest.mark.needs_db
def test_old_runs_keep_their_columns_and_get_null_in_new_ones(db):
    if not db["old"]:
        pytest.skip("DB/runs/ に、版 1 の run が無い")
    con, r = db["con"], db["old"][0]
    for t, cols in OLD_COLS.items():
        src = pd.read_parquet(DB_DIR / "runs" / r / f"{t}.parquet")
        got = con.execute(f"SELECT * FROM {t} WHERE run_id = ?", [r]).df()
        assert len(got) == len(src)
        for c in cols:
            assert sorted(got[c].tolist(), key=str) == sorted(src[c].tolist(), key=str), (t, c)
    assert con.execute("SELECT count(*) FROM ticks WHERE run_id = ? AND est_sum_exposed IS NOT NULL", [r]).fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM episodes WHERE run_id = ? AND n_est IS NOT NULL", [r]).fetchone()[0] == 0
    for t in ("sim_iters", "agent_ticks", "pv_won", "pv_all", "epl"):
        assert con.execute(f"SELECT count(*) FROM {t} WHERE run_id = ?", [r]).fetchone()[0] == 0, t


def test_new_tables_and_views(db):
    con = db["con"]
    n = lambda t, r: con.execute(f"SELECT count(*) FROM {t} WHERE run_id = ?", [r]).fetchone()[0]
    for r in db["new"]:
        assert n("sim_iters", r) >= 96 and n("agent_ticks", r) == 2 * 48 * 48 and n("pv_all", r) > 0 and n("bids_all", r) == n("pv_all", r)
        assert n("epl", r) == 96
    assert n("sim_iters", db["basic"]) >= 48 and n("agent_ticks", db["basic"]) == 0 and n("pv_won", db["basic"]) == 0
    kinds = dict(con.execute("SELECT table_name, table_type FROM information_schema.tables").fetchall())
    assert kinds["sim_iters"] == "BASE TABLE" and all(kinds[v] == "VIEW" for v in ("agent_ticks", "pv_won", "pv_all", "bids_all", "epl", "runs_newest"))
    com = con.execute("SELECT comment FROM duckdb_columns() WHERE table_name = 'ticks' AND column_name = 'est_sum_exposed'").fetchone()[0]
    assert "推定価値" in com and "なぜ取るか" in com


def test_epl_view_satisfies_the_identity(db):
    con = db["con"]
    df = con.execute("SELECT * FROM epl WHERE reward >= 1 AND n_est > 0").df()
    assert len(df) >= 5
    assert ((df.E * df.P * df.L - df.R).abs() / df.R).max() < 1e-9
    assert con.execute("SELECT count(*) FROM epl WHERE reward = 0 AND (L IS NOT NULL OR R IS NOT NULL)").fetchone()[0] == 0


def test_runs_newest_picks_one_run_per_config_and_position(db):
    con = db["con"]
    a = con.execute("SELECT config_key, player_index, count(*) c FROM runs_newest GROUP BY 1, 2").df()
    assert (a.c == 1).all()
    b = con.execute("SELECT count(DISTINCT (config_key, player_index)) FROM runs WHERE status = 'ok' AND github_dirty = false").fetchone()[0]
    assert len(a) == b


def test_config_key_ignores_code_version_and_record_level():
    spec = dict(player=dict(strategy="PID", kwargs={}), episodes=[0, 1], seed=1, gin=None)
    assert build_db.config_key(spec) == build_db.config_key(dict(spec, record="full", name="x", spec_file="y"))
    assert build_db.config_key(spec) != build_db.config_key(dict(spec, seed=2))
