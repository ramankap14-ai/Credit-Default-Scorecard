"""Step 3: WoE/IV scorecard, evaluation, cut-off economics, fairness audit, charts."""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "outputs/charts"
df = pd.read_csv(ROOT / "data/processed/credit_clean.csv")

# SEX, MARRIAGE and EDUCATION are deliberately NOT model inputs (see README: fairness / regulation).
CANDIDATES = ["PAY_1", "PAY_2", "n_late", "max_delay", "util_1", "avg_util", "pay_ratio_1",
              "avg_pay_ratio", "LIMIT_BAL", "AGE"]
train, test = train_test_split(df, test_size=0.30, stratify=df["default"], random_state=42)
train, test = train.copy(), test.copy()
print(f"train {len(train)} / test {len(test)}; default rate {train['default'].mean():.2%} / {test['default'].mean():.2%}")

# ---------- 1. Binning (edges learned on TRAIN only) ----------
def pay_bins(s):
    return pd.cut(s, [-np.inf, -1, 0, 1, 2, np.inf], labels=["<=-1 (paid/none)", "0 (revolving)", "1 month late", "2 months late", "3+ months late"])

def make_binner(name):
    if name in ("PAY_1", "PAY_2"):
        return lambda s: pay_bins(s)
    if name in ("n_late", "max_delay"):
        return lambda s: pd.cut(s, [-np.inf, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"])
    edges = np.unique(np.quantile(train[name], [0, .2, .4, .6, .8, 1]))
    edges[0], edges[-1] = -np.inf, np.inf
    return lambda s: pd.cut(s, edges, include_lowest=True)

binners = {c: make_binner(c) for c in CANDIDATES}

# ---------- 2. Weight of evidence and information value ----------
G, B = (train["default"] == 0).sum(), (train["default"] == 1).sum()
woe_tables, iv = {}, {}
for c in CANDIDATES:
    b = binners[c](train[c])
    t = train.groupby(b, observed=False)["default"].agg(bad="sum", n="count")
    t["good"] = t["n"] - t["bad"]
    t["woe"] = np.log(((t["good"] + 0.5) / G) / ((t["bad"] + 0.5) / B))
    t["iv_part"] = ((t["good"] / G) - (t["bad"] / B)) * t["woe"]
    t["bad_rate"] = t["bad"] / t["n"]
    woe_tables[c] = t
    iv[c] = t["iv_part"].sum()
iv = pd.Series(iv).sort_values(ascending=False)
print("\nInformation value (IV):\n", iv.round(3).to_string())
keep = list(iv[iv >= 0.02].index)
print("kept (IV >= 0.02):", keep)

def to_woe(d):
    out = pd.DataFrame(index=d.index)
    for c in keep:
        b = binners[c](d[c])
        out[c] = b.map(woe_tables[c]["woe"]).astype(float)
    return out

ytr, yte = train["default"], test["default"]

# ---------- 3. Logistic regression on WoE, with sign pruning ----------
# Higher WoE = safer, so every coefficient must be negative. A positive sign means collinearity is
# distorting the variable; the standard fix is to drop the weakest offender and refit.
while True:
    Xtr = to_woe(train)
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    coef = pd.Series(lr.coef_[0], index=keep)
    wrong = coef[coef > 0]
    if wrong.empty:
        break
    drop = iv[wrong.index].idxmin()
    print(f"dropping {drop}: positive coefficient {coef[drop]:.3f} (IV {iv[drop]:.3f})")
    keep.remove(drop)
Xte = to_woe(test)
print("\nFinal variables and coefficients:\n", coef.round(3).to_string())

pd_tr, pd_te = lr.predict_proba(Xtr)[:, 1], lr.predict_proba(Xte)[:, 1]
# ---------- 4. Points: 600 = good:bad odds of 50:1, +20 points doubles the odds ----------
PDO, BASE, ODDS = 20, 600, 50
factor = PDO / np.log(2); offset = BASE - factor * np.log(ODDS)
def score(Xw):
    return offset - factor * (lr.intercept_[0] + Xw.values @ coef.values)
train["score"], test["score"] = score(Xtr), score(Xte)
train["pd"], test["pd"] = pd_tr, pd_te

rows = []
for c in keep:
    t = woe_tables[c].copy()
    t["points"] = -(factor * (coef[c] * t["woe"] + lr.intercept_[0] / len(keep))) + offset / len(keep)
    for lab, r in t.iterrows():
        rows.append({"variable": c, "bin": str(lab), "customers": int(r["n"]), "bad_rate_pct": round(100 * r["bad_rate"], 1),
                     "woe": round(r["woe"], 3), "points": round(r["points"], 1)})
pd.DataFrame(rows).to_csv(ROOT / "outputs/scorecard_points.csv", index=False)
iv.round(4).rename("iv").to_csv(ROOT / "outputs/information_value.csv")

# ---------- 5. Evaluation on the held-out TEST set ----------
def ks(d):
    return stats.ks_2samp(d.loc[d["default"] == 0, "score"], d.loc[d["default"] == 1, "score"]).statistic
auc_tr, auc_te = roc_auc_score(ytr, pd_tr), roc_auc_score(yte, pd_te)
print(f"\nScorecard AUC train {auc_tr:.3f} / test {auc_te:.3f}; Gini test {2*auc_te-1:.3f}; KS test {ks(test):.3f}")
raw = [c for c in df.columns if c not in ("ID", "default", "SEX", "MARRIAGE", "EDUCATION")]
gb = HistGradientBoostingClassifier(random_state=42, max_depth=3, learning_rate=0.05, max_iter=200)
gb.fit(train[raw], ytr); auc_gb = roc_auc_score(yte, gb.predict_proba(test[raw])[:, 1])
print(f"Benchmark gradient boosting (raw features) test AUC {auc_gb:.3f}")
print(f"Latest repayment status alone (PAY_1) test AUC {roc_auc_score(yte, test['PAY_1']):.3f}")

# ---------- 6. Calibration by risk decile (test) ----------
test["risk_decile"] = pd.qcut(test["pd"].rank(method="first", ascending=False), 10, labels=range(1, 11))
cal = test.groupby("risk_decile", observed=True).agg(customers=("default", "size"), predicted_pd=("pd", "mean"), actual_default=("default", "mean")).round(3)
print("\nCalibration (decile 1 = riskiest):\n", cal.to_string())
cal.to_csv(ROOT / "outputs/calibration_by_decile.csv")

# ---------- 7. Cut-off economics ----------
# Unit economics: an approved good account earns 1 unit; an approved bad account loses R units (assumption).
# Break-even default rate = 1 / (1 + R). Cut-off chosen on TRAIN, results reported on TEST.
rates = np.arange(0.1, 1.01, 0.1)
def profit(d, thr, R): 
    a = d[d["score"] >= thr]; bad = a["default"].sum(); good = len(a) - bad
    return good - R * bad, len(a), bad
out = []
for R in [2, 3, 4, 6]:
    best = None
    for r in rates:
        thr = np.quantile(train["score"], 1 - r) if r < 1 else -np.inf
        ptr, _, _ = profit(train, thr, R)
        if best is None or ptr > best[0]:
            best = (ptr, r, thr)
    _, r_best, thr = best
    pt, n_app, bad = profit(test, thr, R)
    pt_all, _, _ = profit(test, -np.inf, R)
    out.append({"loss_to_gain_ratio": R, "break_even_default_rate_pct": round(100 / (1 + R), 1),
                "chosen_approval_rate_pct": round(100 * r_best), "cutoff_score": round(thr, 1),
                "test_approved": n_app, "test_default_rate_among_approved_pct": round(100 * bad / n_app, 1),
                "test_profit_at_cutoff": round(pt), "test_profit_approve_all": round(pt_all)})
econ = pd.DataFrame(out)
print("\nCut-off economics (chosen on train, evaluated on test):\n", econ.to_string(index=False))
econ.to_csv(ROOT / "outputs/cutoff_economics.csv", index=False)

# ---------- 8. Fairness audit (SEX and age were NOT used by the scorecard except AGE) ----------
R4 = econ.loc[econ["loss_to_gain_ratio"] == 4].iloc[0]; thr4 = R4["cutoff_score"]
test["approved"] = (test["score"] >= thr4).astype(int)
test["age_band"] = pd.cut(test["AGE"], [0, 29, 39, 49, 100], labels=["<30", "30-39", "40-49", "50+"])
fa = []
for col in ["SEX", "age_band"]:
    g = test.groupby(col, observed=True).agg(customers=("approved", "size"), approval_rate=("approved", "mean"), default_rate=("default", "mean"))
    g["default_rate_among_approved"] = test[test["approved"] == 1].groupby(col, observed=True)["default"].mean()
    g = g.round(3).reset_index().rename(columns={col: "group"}); g.insert(0, "dimension", col); fa.append(g)
fa = pd.concat(fa); print("\nApproval audit at R=4 cut-off:\n", fa.to_string(index=False))
fa.to_csv(ROOT / "outputs/fairness_audit.csv", index=False)

# ---------- 9. Write scores to SQLite for the SQL files ----------
train["split"], test["split"] = "train", "test"
scored = pd.concat([train, test])[["ID", "split", "score", "pd", "default"]].rename(columns={"default": "defaulted"})
with sqlite3.connect(ROOT / "data/credit.db") as con:
    scored.to_sql("scored", con, if_exists="replace", index=False)

# ---------- 10. Charts ----------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
p1 = df.assign(s=pd.cut(df["PAY_1"], [-3, -1, 0, 1, 2, 10], labels=["Paid/none", "Revolving", "1m late", "2m late", "3m+ late"])).groupby("s", observed=True)["default"].mean() * 100
fig, ax = plt.subplots(figsize=(6, 3.6)); bars = ax.bar(p1.index.astype(str), p1.values, color="#1f4e79")
ax.axhline(df["default"].mean() * 100, color="grey", ls=":", label="Overall 22.1%"); ax.set_ylabel("Default rate next month (%)")
ax.set_title("Latest repayment status drives default"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(CHARTS / "default_rate_by_repayment_status.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.6))
bins = np.linspace(test["score"].quantile(0.005), test["score"].quantile(0.995), 40)
ax.hist(test.loc[test["default"] == 0, "score"], bins=bins, alpha=.65, color="#1f4e79", label="Did not default", density=True)
ax.hist(test.loc[test["default"] == 1, "score"], bins=bins, alpha=.65, color="#c0392b", label="Defaulted", density=True)
ax.set_xlabel("Scorecard points (higher = safer)"); ax.set_ylabel("Density"); ax.set_title(f"Score separation (test KS = {ks(test):.2f})"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(CHARTS / "score_distribution.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.6))
for R, colr in [(2, "#7fb3d5"), (4, "#1f4e79"), (6, "#c0392b")]:
    ys = []
    for r in rates:
        thr = np.quantile(train["score"], 1 - r) if r < 1 else -np.inf
        ys.append(profit(test, thr, R)[0])
    ax.plot(rates * 100, ys, marker="o", color=colr, label=f"Loss = {R}x gain")
ax.axhline(0, color="black", lw=.8); ax.set_xlabel("Approval rate (%) - safest customers first"); ax.set_ylabel("Profit (units, test set)")
ax.set_title("Profit vs approval rate"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(CHARTS / "profit_vs_approval_rate.png", dpi=160); plt.close(fig)
