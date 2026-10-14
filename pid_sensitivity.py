"""Sensitivity of the partial information decomposition to how the panel is represented.

The decomposition uses two sources: sex, and an out-of-fold prediction of
creatinine from the rest of the panel, discretised into quantile bins. The
second source is therefore a COMPRESSED summary of the panel, not the panel
itself. Compression can only discard panel information about creatinine, so
it biases unique-to-sex UPWARD. This script bounds that bias by varying how
much compression is applied: more bins and a stronger regressor both let the
panel source carry more, which should reduce unique-to-sex if the bias matters.
"""
import argparse, json
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
from pid import broja_pid
from run_analysis import discretise, joint_from_samples

SEED = 42
COND = ["age", "urea", "sodium", "potassium", "chloride"]
REGRESSORS = {
    "linear": lambda: LinearRegression(),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=SEED),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--creatinine-units", choices=["umol", "mgdl"], default="umol")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    df.columns = [c.strip().lower() for c in df.columns]
    s = df["sex"].astype(str).str.strip().str.upper()
    df["sex_bin"] = np.where(s.isin(["F", "FEMALE", "1"]), 1, 0)
    df["creat_mgdl"] = (df["creatinine"] / 88.4 if a.creatinine_units == "umol"
                        else df["creatinine"])
    df = df.dropna(subset=["age"] + COND[1:] + ["creat_mgdl"]).reset_index(drop=True)

    creat = df["creat_mgdl"].values
    sex = df["sex_bin"].astype(int).values
    Z = df[COND].values
    cv = KFold(5, shuffle=True, random_state=SEED)

    print(f"\n=== {a.label}  n={len(df)} ===")
    print("unique-to-sex as a percentage of I(sex; creatinine) -- the Abstract's figure\n")
    print(f"{'panel regressor':<20}{'out-of-fold R2':>15}{'bins':>7}"
          f"{'I_total':>10}{'I_sex':>9}{'uniq_sex%':>11}{'uniq_panel%':>13}{'redund%':>9}")

    rows, out = [], {"label": a.label, "n": len(df), "runs": []}
    for rname, mk in REGRESSORS.items():
        pred = cross_val_predict(mk(), Z, creat, cv=cv)
        r2 = r2_score(creat, pred)
        for bins in (3, 4, 5, 6, 8):
            Y = discretise(creat, bins)
            X2 = discretise(pred, bins)
            r = broja_pid(joint_from_samples(Y, sex, X2), n_restarts=6, seed=SEED)
            tot, isex = r["I_total"], r["I_x1"]
            u = 100 * r["unique_x1"] / isex if isex > 1e-12 else float("nan")
            up = 100 * r["unique_x2"] / tot if tot > 1e-12 else float("nan")
            rd = 100 * r["redundant"] / tot if tot > 1e-12 else float("nan")
            print(f"{rname:<20}{r2:>15.3f}{bins:>7}{tot:>10.4f}{isex:>9.4f}"
                  f"{u:>11.1f}{up:>13.1f}{rd:>9.1f}")
            rows.append(u)
            out["runs"].append({"regressor": rname, "oof_r2": float(r2), "bins": bins,
                                "I_total": r["I_total"], "I_sex": r["I_x1"],
                                "pct_unique_sex_of_sex_info": float(u),
                                "pct_unique_panel_of_joint": float(up),
                                "pct_redundant_of_joint": float(rd)})

    rows = np.array(rows, float)
    print(f"\nacross all {len(rows)} configurations: "
          f"unique-to-sex {rows.min():.1f}% to {rows.max():.1f}% "
          f"(median {np.median(rows):.1f}%)")
    out["summary"] = {"min": float(rows.min()), "max": float(rows.max()),
                      "median": float(np.median(rows))}
    json.dump(out, open(f"pid_sensitivity_{a.label}.json", "w"), indent=1)
    print(f"wrote pid_sensitivity_{a.label}.json")


if __name__ == "__main__":
    main()
