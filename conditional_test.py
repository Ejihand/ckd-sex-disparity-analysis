"""Conditional-independence test for I(creatinine ; sex | panel).

The original analysis permuted the sex label globally. That breaks the
sex-panel relationship as well as the sex-creatinine relationship, so it
tests the joint null  sex _||_ (creatinine, Z)  rather than the conditional
null  sex _||_ creatinine | Z.  Because sex is genuinely associated with
urea and the electrolytes, the global null distribution sits too low and the
test is anti-conservative.

This replaces it with the Conditional Randomization Test (Candes, Fan,
Janson & Lv 2018, JRSS-B 80:551-577): estimate P(sex = 1 | Z) by
cross-fitting, then draw synthetic sex labels from that fitted conditional
distribution.  Each draw preserves the sex-Z relationship and destroys only
the sex-creatinine association given Z, which is exactly the intended null.

A restricted within-stratum permutation is run alongside it as a second,
model-lighter check (Berrett, Wang, Barber & Samworth 2020, JRSS-B 82:175-197).

Usage:  python conditional_test.py --data cohort_clean.csv --label nigerian
"""
import argparse, json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from infotheory import gc_cmi

SEED = 42
COND = ["age", "urea", "sodium", "potassium", "chloride"]


def propensity(Z, sex, kind="gb", seed=SEED):
    """Cross-fitted P(sex = 1 | Z). Cross-fitting keeps each record's fitted
    probability out of its own training fold."""
    mk = (GradientBoostingClassifier(random_state=seed) if kind == "gb"
          else LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(mk, Z, sex, cv=cv, method="predict_proba")[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6)


def crt(creat, sex, Z, p_hat, n_perm=500, rng=None):
    """Conditional randomization test: draw sex* ~ Bernoulli(p_hat(Z))."""
    rng = np.random.default_rng(SEED) if rng is None else rng
    obs = gc_cmi(creat, sex, Z)
    null = np.empty(n_perm)
    for b in range(n_perm):
        null[b] = gc_cmi(creat, (rng.random(len(p_hat)) < p_hat).astype(float), Z)
    return obs, (np.sum(null >= obs) + 1) / (n_perm + 1), null


def stratified_perm(creat, sex, Z, p_hat, n_perm=500, n_strata=20, rng=None):
    """Permute sex only within strata of the fitted propensity, which holds
    the sex-Z relationship approximately fixed without assuming the model is
    exactly right."""
    rng = np.random.default_rng(SEED) if rng is None else rng
    obs = gc_cmi(creat, sex, Z)
    edges = np.quantile(p_hat, np.linspace(0, 1, n_strata + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    strata = np.digitize(p_hat, edges[1:-1])
    idx = [np.where(strata == s)[0] for s in range(n_strata)]
    null = np.empty(n_perm)
    for b in range(n_perm):
        sx = sex.copy()
        for ix in idx:
            if len(ix) > 1:
                sx[ix] = sex[rng.permutation(ix)]
        null[b] = gc_cmi(creat, sx, Z)
    return obs, (np.sum(null >= obs) + 1) / (n_perm + 1), null


def global_perm(creat, sex, Z, n_perm=500, rng=None):
    """The original procedure, reproduced for comparison."""
    rng = np.random.default_rng(SEED) if rng is None else rng
    obs = gc_cmi(creat, sex, Z)
    null = np.array([gc_cmi(creat, rng.permutation(sex), Z) for _ in range(n_perm)])
    return obs, (np.sum(null >= obs) + 1) / (n_perm + 1), null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--creatinine-units", choices=["umol", "mgdl"], default="umol")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    df.columns = [c.strip().lower() for c in df.columns]
    s = df["sex"].astype(str).str.strip().str.upper()
    df["sex_bin"] = np.where(s.isin(["F", "FEMALE", "1"]), 1.0, 0.0)
    df["creat_mgdl"] = (df["creatinine"] / 88.4 if a.creatinine_units == "umol"
                        else df["creatinine"])
    df = df.dropna(subset=["age"] + COND[1:] + ["creat_mgdl", "sex_bin"]).reset_index(drop=True)

    creat = df["creat_mgdl"].values
    sex = df["sex_bin"].values
    Z = df[COND].values
    rng = np.random.default_rng(SEED)

    print(f"\n=== {a.label}  n={len(df)}  female={int(sex.sum())} ===")

    p_gb = propensity(Z, sex, "gb")
    p_lr = propensity(Z, sex, "lr")
    auc_gb = roc_auc_score(sex, p_gb)
    auc_lr = roc_auc_score(sex, p_lr)
    print(f"P(sex|Z) cross-fitted AUC: gradient boosting {auc_gb:.3f}, "
          f"logistic {auc_lr:.3f}")
    print("  (the closer this is to 0.5, the less the two nulls can differ:")
    print("   a global permutation is wrong in principle but harmless in practice")
    print("   when sex is only weakly predictable from the conditioning panel)\n")

    res = {"label": a.label, "n": len(df), "n_female": int(sex.sum()),
           "propensity_auc_gb": auc_gb, "propensity_auc_lr": auc_lr,
           "n_perm": a.n_perm}

    rows = []
    for name, fn, kw in (
            ("global permutation (original)", global_perm, {}),
            ("CRT, gradient-boosting propensity", crt, {"p_hat": p_gb}),
            ("CRT, logistic propensity", crt, {"p_hat": p_lr}),
            ("stratified permutation (20 strata)", stratified_perm, {"p_hat": p_gb})):
        obs, p, null = fn(creat, sex, Z, n_perm=a.n_perm, rng=rng, **kw)
        rows.append((name, obs, null.mean(), np.quantile(null, 0.95), p))
        res[name] = {"observed": float(obs), "null_mean": float(null.mean()),
                     "null_q95": float(np.quantile(null, 0.95)), "p": float(p)}

    print(f"{'procedure':<36}{'observed':>10}{'null mean':>11}{'null q95':>10}{'p':>9}")
    for name, obs, nm, q95, p in rows:
        print(f"{name:<36}{obs:>10.4f}{nm:>11.4f}{q95:>10.4f}{p:>9.4f}")

    out = a.out or f"conditional_test_{a.label}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
