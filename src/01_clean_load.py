"""Step 1: clean the raw data, engineer features, load into SQLite."""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/raw/UCI_Credit_Card.csv")
n0 = len(df)

df = df.rename(columns={"default.payment.next.month": "default", "PAY_0": "PAY_1"})

# Undocumented category codes -> "other" (documented in README)
log = {"education_undocumented_0_5_6": int(df["EDUCATION"].isin([0, 5, 6]).sum()),
       "marriage_undocumented_0": int((df["MARRIAGE"] == 0).sum())}
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

bill = [f"BILL_AMT{i}" for i in range(1, 7)]
pay_amt = [f"PAY_AMT{i}" for i in range(1, 7)]
pay_st = [f"PAY_{i}" for i in range(1, 7)]

# --- Features: all built only from the 6 months BEFORE the default month ---
df["util_1"] = (df["BILL_AMT1"] / df["LIMIT_BAL"]).clip(lower=0)             # latest utilisation
df["avg_util"] = (df[bill].clip(lower=0).mean(axis=1) / df["LIMIT_BAL"])     # 6-month average utilisation
df["pay_ratio_1"] = np.where(df["BILL_AMT1"] > 0, df["PAY_AMT1"] / df["BILL_AMT1"], 1.0)
df["pay_ratio_1"] = df["pay_ratio_1"].clip(0, 1)                             # share of latest bill paid
tot_bill = df[bill].clip(lower=0).sum(axis=1)
df["avg_pay_ratio"] = np.where(tot_bill > 0, df[pay_amt].sum(axis=1) / tot_bill, 1.0)
df["avg_pay_ratio"] = df["avg_pay_ratio"].clip(0, 1)
df["n_late"] = (df[pay_st] >= 1).sum(axis=1)                                 # months with a payment delay
df["max_delay"] = df[pay_st].clip(lower=0).max(axis=1)                       # worst delay in months

df.to_csv(ROOT / "data/processed/credit_clean.csv", index=False)
with sqlite3.connect(ROOT / "data/credit.db") as con:
    df.rename(columns={"default": "defaulted"}).to_sql("customers", con, if_exists="replace", index=False)  # DEFAULT is an SQL keyword

print(f"rows: {n0} (none dropped); columns: {df.shape[1]}")
print("recoded:", log)
print(f"default rate: {df['default'].mean():.2%} ({int(df['default'].sum())} of {len(df)})")
print(f"negative latest bill amounts (kept; credit balances): {int((df['BILL_AMT1'] < 0).sum())}")
