"""Step 4: run the SQL files against the scored database and save the result tables."""
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
with sqlite3.connect(ROOT / "data/credit.db") as con:
    for sql_file, out in [("01_segment_default_rate", "segment_default_rate"),
                          ("02_score_bands", "score_bands_test"),
                          ("03_cutoff_by_depth", "cutoff_by_depth_test")]:
        d = pd.read_sql_query((ROOT / "sql" / f"{sql_file}.sql").read_text(), con)
        d.to_csv(ROOT / f"outputs/{out}.csv", index=False)
        print(f"\n== {out} ==\n{d.to_string(index=False)}")
