"""DB に SQL を投げて表示する（Claude が使う想定）。

    .venv/Scripts/python.exe DB/query.py "SELECT * FROM run_summary"
    .venv/Scripts/python.exe DB/query.py "SELECT ..." --csv out.csv
    .venv/Scripts/python.exe DB/query.py --describe          # 表・列の意味（COMMENT）を一覧
"""
import argparse
from pathlib import Path

import duckdb
import pandas as pd

DB = Path(__file__).resolve().parent / "auctionnet.duckdb"

ap = argparse.ArgumentParser()
ap.add_argument("sql", nargs="?")
ap.add_argument("--csv")
ap.add_argument("--describe", action="store_true")
a = ap.parse_args()
con = duckdb.connect(str(DB), read_only=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 80)
if a.describe:
    print(con.execute("SELECT table_name, comment FROM duckdb_tables()").df().to_string())
    print(con.execute("SELECT table_name, column_name, comment FROM duckdb_columns() WHERE comment IS NOT NULL").df().to_string())
else:
    df = con.execute(a.sql).df()
    if a.csv:
        df.to_csv(a.csv, index=False, encoding="utf-8")
        print("wrote", a.csv, len(df))
    else:
        print(df.to_string())
