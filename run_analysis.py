"""
Paper 2 analysis pipeline — the sex-proxy information decomposition.

Runs Experiments 1-4 of the concept note on the study cohort and writes a
results file that maps one-to-one onto the manuscript's Results tables.

Usage
-----
    python run_analysis.py --data cohort.csv --out results.json

Expected columns (rename via the --map option if yours differ):
    age, sex, creatinine, urea, sodium, potassium, chloride
`sex` may be coded as 0/1, M/F, or Male/Female.

IMPORTANT
---------
Creatinine is expected in umol/L (as in the source cohort). The MDRD derivation
converts internally. If your file is already in mg/dL pass --creatinine-units mgdl.

Nothing in this script invents values: every number written to the results file
is computed from the supplied data.
"""
import argparse, json, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

from infotheory import (ksg_cmi, mixed_mi, bootstrap_ci, permutation_test,
                        gc_cmi, gc_mi)
from pid import broja_pid, joint_from_samples, discretise

K = 5
N_BOOT = 500
N_PERM = 500
SEED = 42

ANALYTES = ["creatinine", "urea", "sodium", "potassium", "chloride"]


# ----------------------------------------------------------------------
def load(path, creat_units="umol"):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in ["age", "sex"] + ANALYTES if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing required columns: {missing}\nFound: {list(df.columns)}")

    s = df["sex"].astype(str).str.strip().str.upper()
    df["sex_bin"] = np.where(s.isin(["F", "FEMALE", "1"]), 1,
                      np.where(s.isin(["M", "MALE", "0"]), 0, np.nan))
    if df["sex_bin"].isna().any():
        sys.exit(f"ERROR: unparsed sex values: {sorted(set(s[df['sex_bin'].isna()]))}")

    df["creat_mgdl"] = df["creatinine"] / 88.4 if creat_units == "umol" else df["creatinine"]
    df = df.dropna(subset=["age"] + ANALYTES + ["sex_bin"]).reset_index(drop=True)
    return df


def mdrd(creat_mgdl, age, female, apply_sex_factor=True):
    """Race-free 4-variable MDRD. apply_sex_factor=False sets the 0.742 multiplier to 1."""
    f = np.where(female == 1, 0.742, 1.0) if apply_sex_factor else 1.0
    return 175 * np.power(creat_mgdl, -1.154) * np.power(age, -0.203) * f


def ckdepi2021(creat_mgdl, age, female, apply_sex_factor=True):
    """
    CKD-EPI 2021 creatinine equation (race-free), National Kidney Foundation:

        eGFR = 142 * min(Scr/k,1)^a * max(Scr/k,1)^-1.200 * 0.9938^age * 1.012 [if female]
        k = 0.7 (female) / 0.9 (male);  a = -0.241 (female) / -0.302 (male)

    Unlike MDRD the sex adjustment is three-part: sex-specific k and a plus the 1.012
    multiplier. apply_sex_factor=False therefore applies the MALE k and a to every record
    and drops the multiplier, which is the counterfactual "equation with no sex adjustment".
    """
    if apply_sex_factor:
        k = np.where(female == 1, 0.7, 0.9)
        a = np.where(female == 1, -0.241, -0.302)
        mult = np.where(female == 1, 1.012, 1.0)
    else:
        k = np.full_like(np.asarray(creat_mgdl, dtype=float), 0.9)
        a = np.full_like(np.asarray(creat_mgdl, dtype=float), -0.302)
        mult = 1.0
    r = np.asarray(creat_mgdl, dtype=float) / k
    return 142 * np.power(np.minimum(r, 1.0), a) * np.power(np.maximum(r, 1.0), -1.200) \
           * np.power(0.9938, np.asarray(age, dtype=float)) * mult


EQUATIONS = {"mdrd": mdrd, "ckdepi": ckdepi2021}
EQ = mdrd   # set from --equation in main()


def kdigo(egfr):
    return np.select(
        [egfr >= 90, egfr >= 60, egfr >= 30, egfr >= 15],
        [1, 2, 3, 4], default=5).astype(int)


# ----------------------------------------------------------------------
def experiment_1(df, rng):
    """Sex-aware vs sex-blind predictive ceiling (anchors the ~0.97 claim)."""
    try:
        from xgboost import XGBClassifier
        mk = lambda: XGBClassifier(max_depth=4, learning_rate=0.08, subsample=0.80,
                                   reg_lambda=1.50, n_estimators=300, tree_method="hist",
                                   objective="multi:softprob", num_class=5,
                                   random_state=SEED, verbosity=0)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        mk = lambda: GradientBoostingClassifier(max_depth=4, learning_rate=0.08,
                                                random_state=SEED)
        print("  [warn] xgboost unavailable; using sklearn GradientBoosting")

    y = kdigo(EQ(df["creat_mgdl"].values, df["age"].values, df["sex_bin"].values)) - 1
    blind = df[["age"] + ANALYTES].values
    aware = np.column_stack([blind, df["sex_bin"].values])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out = {}
    for name, X in (("sex_blind", blind), ("sex_aware", aware)):
        acc = cross_val_score(mk(), X, y, cv=cv, scoring="accuracy")
        f1 = cross_val_score(mk(), X, y, cv=cv, scoring="f1_macro")
        out[name] = {"cv_accuracy_mean": float(acc.mean()), "cv_accuracy_sd": float(acc.std()),
                     "cv_macro_f1_mean": float(f1.mean()), "cv_macro_f1_sd": float(f1.std()),
                     "folds": [float(a) for a in acc]}
    out["ceiling_gap_accuracy"] = out["sex_aware"]["cv_accuracy_mean"] - out["sex_blind"]["cv_accuracy_mean"]
    return out


COND = ["age", "urea", "sodium", "potassium", "chloride"]


def experiment_2(df, rng):
    """
    I(creatinine ; sex | age, urea, Na, K, Cl) -- the BIOLOGICAL sex signal.

    PRIMARY estimator is the Gaussian copula (gc_cmi). The k-NN (KSG) estimator is
    reported as a secondary sensitivity analysis ONLY at low conditioning dimension:
    validation on synthetic data showed that KSG collapses toward zero as the
    conditioning set grows past two variables at this sample size, even when the
    conditioning variables are pure noise. Using KSG at full conditioning dimension
    would manufacture a spurious "sex is redundant" result. See METHODS_NOTE.md.
    """
    creat = df["creat_mgdl"].values
    sex = df["sex_bin"].values
    Z = df[COND].values

    # ---- primary: Gaussian copula ----
    point_gc = gc_cmi(creat, sex, Z)
    mean_gc, lo_gc, hi_gc = bootstrap_ci(lambda a, b, c: gc_cmi(a, b, c),
                                         [creat, sex, Z], n_boot=N_BOOT, rng=rng)
    _, p_gc, null_gc = permutation_test(lambda a, b, c: gc_cmi(a, b, c),
                                        [creat, sex, Z], shuffle_index=1,
                                        n_perm=N_PERM, rng=rng)
    marginal_gc = gc_mi(creat, sex)

    # ---- dimension-sensitivity diagnostic (reportable result, both estimators) ----
    dim_sweep = {}
    for d_ in range(1, len(COND) + 1):
        Zi = df[COND[:d_]].values
        dim_sweep[str(d_)] = {"gaussian_copula": gc_cmi(creat, sex, Zi),
                              "ksg": ksg_cmi(creat, sex, Zi, k=K, rng=rng)}

    # ---- secondary: KSG, low conditioning dimension only ----
    ksg_low = ksg_cmi(creat, sex, df[COND[:2]].values, k=K, rng=rng)

    return {
        "primary_estimator": "gaussian_copula",
        "cmi_creatinine_sex_given_panel_nats": point_gc,
        "bootstrap_mean": mean_gc, "ci95_low": lo_gc, "ci95_high": hi_gc,
        "permutation_p": p_gc, "permutation_null_mean": null_gc,
        "marginal_mi_creatinine_sex_nats": marginal_gc,
        "redundancy_fraction": float(1 - point_gc / marginal_gc) if marginal_gc > 1e-9 else None,
        "dimension_sensitivity": dim_sweep,
        "ksg_secondary_2cond": ksg_low,
        "ksg_k_sensitivity_2cond": {str(kk): ksg_cmi(creat, sex, df[COND[:2]].values,
                                                     k=kk, rng=rng) for kk in (3, 5, 10, 20)},
    }


def experiment_3(df, rng, bins=4):
    """
    BROJA partial information decomposition (RQ2), in bits.

    Sources: sex (binary) and what the rest of the panel knows about creatinine,
    represented by an out-of-fold prediction of creatinine from age, urea, Na, K, Cl.
    Target: measured creatinine. Both target and panel source are discretised into
    quantile bins so the decomposition is defined on a finite alphabet.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict

    creat = df["creat_mgdl"].values
    sex = df["sex_bin"].astype(int).values
    pred = cross_val_predict(GradientBoostingRegressor(random_state=SEED),
                             df[COND].values, creat, cv=5)
    Y = discretise(creat, bins)
    X2 = discretise(pred, bins)
    r = broja_pid(joint_from_samples(Y, sex, X2), n_restarts=6, seed=SEED)
    tot = r["I_total"] if r["I_total"] > 1e-12 else float("nan")
    r["pct_unique_sex"] = 100 * r["unique_x1"] / tot
    r["pct_unique_panel"] = 100 * r["unique_x2"] / tot
    r["pct_redundant"] = 100 * r["redundant"] / tot
    r["pct_synergistic"] = 100 * r["synergistic"] / tot
    r["sex_information_that_is_unique_pct"] = 100 * r["unique_x1"] / r["I_x1"] if r["I_x1"] > 1e-12 else None
    r["bins"] = bins
    r["units"] = "bits"
    return r


def experiment_4(df, rng):
    """Equation-manufactured signal: compare true-label vs unity-multiplier label."""
    sex = df["sex_bin"].values
    Z = df[COND].values

    y_true = kdigo(EQ(df["creat_mgdl"].values, df["age"].values, sex, True)).astype(float)
    y_unity = kdigo(EQ(df["creat_mgdl"].values, df["age"].values, sex, False)).astype(float)

    cmi_true = gc_cmi(y_true, sex, Z)
    cmi_unity = gc_cmi(y_unity, sex, Z)
    n_reclass = int(np.sum(y_true != y_unity))

    return {"cmi_stage_sex_given_panel_nats": cmi_true,
            "cmi_stage_unity_multiplier_nats": cmi_unity,
            "equation_manufactured_component_nats": cmi_true - cmi_unity,
            "records_reclassified_by_sex_factor": n_reclass,
            "pct_reclassified": float(100 * n_reclass / len(df))}


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--creatinine-units", choices=["umol", "mgdl"], default="umol")
    ap.add_argument("--skip-exp1", action="store_true", help="skip model fitting")
    ap.add_argument("--equation", choices=["mdrd", "ckdepi"], default="mdrd")
    a = ap.parse_args()

    global EQ
    EQ = EQUATIONS[a.equation]
    rng = np.random.default_rng(SEED)
    df = load(a.data, a.creatinine_units)
    print(f"Loaded n={len(df)}  female={int(df.sex_bin.sum())}  male={int((1-df.sex_bin).sum())}")

    res = {"equation": a.equation, "n": len(df), "n_female": int(df.sex_bin.sum()), "n_male": int((1 - df.sex_bin).sum()),
           "seed": SEED, "k": K, "n_boot": N_BOOT, "n_perm": N_PERM, "units": "nats"}

    if not a.skip_exp1:
        print("Experiment 1: predictive ceiling ...")
        res["experiment_1_ceiling"] = experiment_1(df, rng)
    print("Experiment 2: biological sex signal (CMI on measured creatinine) ...")
    res["experiment_2_biological"] = experiment_2(df, rng)
    print("Experiment 3: partial information decomposition (BROJA) ...")
    res["experiment_3_pid"] = experiment_3(df, rng)
    print("Experiment 4: equation-manufactured signal ...")
    res["experiment_4_equation"] = experiment_4(df, rng)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nWrote {a.out}")
    print("\n--- headline quantities ---")
    e2 = res["experiment_2_biological"]
    print(f"Biological  I(creat;sex|panel) = {e2['cmi_creatinine_sex_given_panel_nats']:.4f} nats "
          f"[{e2['ci95_low']:.4f}, {e2['ci95_high']:.4f}], permutation p={e2['permutation_p']:.4f}")
    e4 = res["experiment_4_equation"]
    print(f"Equation-manufactured component = {e4['equation_manufactured_component_nats']:.4f} nats")
    print(f"Records restaged by the 0.742 factor = {e4['records_reclassified_by_sex_factor']} "
          f"({e4['pct_reclassified']:.1f}%)")
    e3 = res["experiment_3_pid"]
    print(f"PID (bits): unique-sex={e3['unique_x1']:.4f} ({e3['pct_unique_sex']:.1f}%)  "
          f"unique-panel={e3['unique_x2']:.4f} ({e3['pct_unique_panel']:.1f}%)  "
          f"redundant={e3['redundant']:.4f} ({e3['pct_redundant']:.1f}%)  "
          f"synergy={e3['synergistic']:.4f} ({e3['pct_synergistic']:.1f}%)")


if __name__ == "__main__":
    main()
