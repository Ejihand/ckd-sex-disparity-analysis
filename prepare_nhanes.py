"""Build the NHANES analytic CSV the information scripts expect.

Reads the published NHANES 2017-2018 extract and writes nhanes_clean.csv with
the column names and units the rest of the pipeline uses (creatinine in mg/dL).

    python prepare_nhanes.py --xlsx NHANES_Real_CKD_dataset.xlsx
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", required=True)
ap.add_argument("--sheet", default="Analytic_cohort_MDRD_staged")
ap.add_argument("--out", default="nhanes_clean.csv")
a = ap.parse_args()

d = pd.ExcelFile(a.xlsx).parse(a.sheet)
out = pd.DataFrame({
    "age": d["Age"], "sex": d["Gender"], "creatinine": d["Creatinine_mgdl"],
    "urea": d["Urea_BUN"], "sodium": d["Sodium"],
    "potassium": d["Potassium"], "chloride": d["Chloride"],
})
out.to_csv(a.out, index=False)
print(f"wrote {a.out}: n={len(out)}, "
      f"female={(out.sex.str.lower()=='female').sum()}")
