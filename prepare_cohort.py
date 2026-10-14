"""Raw workbook -> cleaned analytic cohort (reproduces the companion study's n = 2,463)."""
import argparse, pandas as pd
from sklearn.ensemble import IsolationForest

ANALYTES = ["Creatinine", "Urea", "Sodium", "Potassium", "Chloride"]

ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", required=True)
ap.add_argument("--out", default="cohort_clean.csv")
ap.add_argument("--contamination", type=float, default=0.015)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

df = pd.read_excel(a.xlsx)
n_raw = len(df)
iso = IsolationForest(contamination=a.contamination, random_state=a.seed)
df = df[iso.fit_predict(df[ANALYTES].values) == 1]
n_iso = len(df)
df = df[df["Age"] >= 18].reset_index(drop=True)

df.columns = [c.lower() for c in df.columns]
df = df.rename(columns={"gender": "sex"})
df.to_csv(a.out, index=False)
print(f"raw={n_raw}  after IsolationForest={n_iso} ({100*n_iso/n_raw:.2f}%)  "
      f"after paediatric exclusion={len(df)}")
print(f"female={(df.sex.str.lower()=='female').sum()}  male={(df.sex.str.lower()=='male').sum()}")
print(f"wrote {a.out}")
