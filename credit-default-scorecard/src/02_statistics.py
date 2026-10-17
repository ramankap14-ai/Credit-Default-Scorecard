"""Step 2: statistical tests on who defaults."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/processed/credit_clean.csv")
df["any_late"] = (df["n_late"] > 0).astype(int)
df["util_high"] = (df["util_1"] > 0.8).astype(int)
rows = []

def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    return np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))

# Chi-square: categorical / binary predictors vs default
for col in ["any_late", "util_high", "SEX", "EDUCATION", "MARRIAGE"]:
    ct = pd.crosstab(df[col], df["default"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    rows.append({"test": "chi-square", "variable": col, "statistic": chi2, "p_value": p,
                 "effect_size": cramers_v(ct), "effect_metric": "Cramer's V"})

# Mann-Whitney: numeric predictors, defaulters vs non-defaulters
for col in ["LIMIT_BAL", "AGE", "util_1", "avg_util", "pay_ratio_1", "avg_pay_ratio", "max_delay"]:
    a = df.loc[df["default"] == 1, col]; b = df.loc[df["default"] == 0, col]
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rows.append({"test": "mann-whitney", "variable": col, "statistic": u, "p_value": p,
                 "effect_size": u / (len(a) * len(b)), "effect_metric": "P(defaulter > non-defaulter)"})

res = pd.DataFrame(rows)
res["p_bonferroni"] = (res["p_value"] * len(res)).clip(upper=1)
res["significant_at_5pct"] = res["p_bonferroni"] < 0.05
res = res.round(4)
res.to_csv(ROOT / "outputs/statistical_tests.csv", index=False)
print(res.to_string(index=False))

# Bootstrap CI: default-rate gap, customers with any late payment vs none
rng = np.random.default_rng(42)
g1 = df.loc[df["any_late"] == 1, "default"].values
g0 = df.loc[df["any_late"] == 0, "default"].values
diffs = [rng.choice(g1, len(g1)).mean() - rng.choice(g0, len(g0)).mean() for _ in range(5000)]
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"\nAny late payment: {g1.mean():.1%} default (n={len(g1)}) vs none: {g0.mean():.1%} (n={len(g0)})")
print(f"Gap = {g1.mean()-g0.mean():.1%}, 95% bootstrap CI [{lo:.1%}, {hi:.1%}]")

# Default rate by latest repayment status (the key monotonic pattern)
print("\nDefault rate by PAY_1 (latest repayment status):")
print(df.groupby("PAY_1")["default"].agg(["mean", "count"]).round(3).to_string())
print("\nDefault rate by SEX:", df.groupby("SEX")["default"].mean().round(3).to_dict())
