"""Answer the three open pipeline questions, on the reproduced pipeline.

Q1  what do the five seeds vary?        -> run all three variants, compare to Table 4.10
Q2  does bootstrap repeat threshold selection? -> run both variants, compare to Table 4.9
Q3  was calibration assessed?           -> compute it now (Brier, ECE, reliability, by sex)
"""
import json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from scipy.stats import fisher_exact
from repro_paper1 import (load, fit, reweigh, inv_freq, adv_prob,
                          apply_thresholds, fairness, FEATS)

N_EST = 300          # recovered: reproduces every published Model A figure
TAU_M = 0.50
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- helpers
def split(df, seed):
    return train_test_split(df, test_size=0.20, random_state=seed,
                            stratify=df.stage)


def select_tau_f(Xtr, ytr, sx_tr, seed, n_est=N_EST, tau_m=TAU_M):
    """Group-conditional female threshold chosen by 5-fold CV on TRAIN only,
    minimising the equalized-odds difference (dissertation S3.9.2)."""
    grid = np.concatenate([[0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05],
                           np.arange(0.075, 0.55, 0.025)])
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    eo = np.zeros(len(grid))
    for itr, iva in skf.split(Xtr, ytr):
        sw = inv_freq(ytr[itr]) * reweigh(sx_tr[itr], ytr[itr])
        m = fit(Xtr[itr], ytr[itr], sw, n_est, seed)
        P = m.predict_proba(Xtr[iva])
        for j, t in enumerate(grid):
            p = apply_thresholds(P, sx_tr[iva], t, tau_m)
            eo[j] += fairness(ytr[iva], p, sx_tr[iva])["eo"]
    return float(grid[eo.argmin()]), float(eo.min() / 5)


def pack(y, p, P, sx):
    return dict(acc=accuracy_score(y, p),
                mf1=f1_score(y, p, average="macro", zero_division=0),
                mrec=recall_score(y, p, average="macro", zero_division=0),
                auc=roc_auc_score(y, P, multi_class="ovr", average="macro"),
                **fairness(y, p, sx))


def full_pipeline(df, seed, tau_f=None):
    """One complete pass: split -> fit A and B -> select tau_f -> evaluate."""
    tr, te = split(df, seed)
    Xtr, ytr, sx_tr = tr[FEATS].values, tr.stage.values, tr.sex.values
    Xte, yte, sx_te = te[FEATS].values, te.stage.values, te.sex.values
    A = fit(Xtr, ytr, None, N_EST, seed)
    swB = inv_freq(ytr) * reweigh(sx_tr, ytr)
    B = fit(Xtr, ytr, swB, N_EST, seed)
    if tau_f is None:
        tau_f, _ = select_tau_f(Xtr, ytr, sx_tr, seed)
    PA, PB = A.predict_proba(Xte), B.predict_proba(Xte)
    pA, pB = PA.argmax(1), apply_thresholds(PB, sx_te, tau_f, TAU_M)
    return dict(tau_f=tau_f, n_test=len(te),
                A=pack(yte, pA, PA, sx_te), B=pack(yte, pB, PB, sx_te),
                yte=yte, sx_te=sx_te, PA=PA, PB=PB, pA=pA, pB=pB)


# ------------------------------------------------------- Q3 calibration
def ece(y_bin, p, bins=10):
    """Expected calibration error, equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, obs = p[m].mean(), y_bin[m].mean()
        e += m.mean() * abs(conf - obs)
        rows.append(dict(bin=f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                         n=int(m.sum()), conf=round(float(conf), 4),
                         observed=round(float(obs), 4)))
    return float(e), rows


def calibration(r):
    """Advanced-stage (G4/G5) risk calibration, overall and by sex."""
    y = (r["yte"] >= 3).astype(float)
    out = {}
    for name in ("A", "B"):
        p = adv_prob(r["P" + name])
        d = {}
        for grp in ("overall", "female", "male"):
            m = np.ones(len(y), bool) if grp == "overall" else r["sx_te"] == grp
            e, rows = ece(y[m], p[m])
            d[grp] = dict(n=int(m.sum()), prevalence=round(float(y[m].mean()), 4),
                          mean_pred=round(float(p[m].mean()), 4),
                          brier=round(float(np.mean((p[m] - y[m]) ** 2)), 4),
                          ece=round(e, 4),
                          calib_in_large=round(float(p[m].mean() - y[m].mean()), 4))
            if grp == "overall":
                d["reliability"] = rows
        out[name] = d
    return out


# -------------------------------------------------------- Q2 bootstrap
def bootstrap(r, df, seed=42, B=2000, refit=False):
    """refit=False  : resample the fixed test predictions (what Table 4.9 implies)
       refit=True   : resample test set AND re-select tau_f each replicate"""
    y, sx, PA, PB, n = r["yte"], r["sx_te"], r["PA"], r["PB"], r["n_test"]
    rng = np.random.default_rng(seed)
    keys = ("fFNR", "EO", "DIR")
    acc = {k: [] for k in keys}
    tr, _ = split(df, 42)
    Xtr, ytr, sx_tr = tr[FEATS].values, tr.stage.values, tr.sex.values
    for b in range(B):
        i = rng.integers(0, n, n)
        if refit and b < 200:        # expensive; 200 replicates
            tf, _ = select_tau_f(Xtr, ytr, sx_tr, int(rng.integers(1e6)))
        else:
            tf = r["tau_f"]
        pB = apply_thresholds(PB[i], sx[i], tf, TAU_M)
        f = fairness(y[i], pB, sx[i])
        acc["fFNR"].append(f["female"]["fnr"])
        acc["EO"].append(f["eo"])
        acc["DIR"].append(f["dir"])
        if refit and b >= 199:
            break
    ci = {}
    for k, v in acc.items():
        v = np.asarray(v, float); v = v[~np.isnan(v)]
        ci[k] = dict(n_rep=len(v), mean=round(float(v.mean()), 4),
                     lo=round(float(np.percentile(v, 2.5)), 4),
                     hi=round(float(np.percentile(v, 97.5)), 4))
    # Model A too, fixed-prediction variant only
    if not refit:
        accA = {k: [] for k in keys}
        rng = np.random.default_rng(seed)
        for _ in range(B):
            i = rng.integers(0, n, n)
            f = fairness(y[i], PA[i].argmax(1), sx[i])
            accA["fFNR"].append(f["female"]["fnr"])
            accA["EO"].append(f["eo"]); accA["DIR"].append(f["dir"])
        ciA = {}
        for k, v in accA.items():
            v = np.asarray(v, float); v = v[~np.isnan(v)]
            ciA[k] = dict(n_rep=len(v), mean=round(float(v.mean()), 4),
                          lo=round(float(np.percentile(v, 2.5)), 4),
                          hi=round(float(np.percentile(v, 97.5)), 4))
        return dict(A=ciA, B=ci)
    return dict(B=ci)


# ------------------------------------------------------------ Q1 seeds
def seed_variants(df, seeds=(42, 1, 7, 123, 2024)):
    """Three readings of 'five independent random seeds'."""
    out = {}

    # V1: full pipeline reseeded -- split, fit and threshold all move
    v1 = [full_pipeline(df, s) for s in seeds]

    # V2: test partition held fixed at seed 42; only model init / threshold move
    tr, te = split(df, 42)
    Xtr, ytr, sx_tr = tr[FEATS].values, tr.stage.values, tr.sex.values
    Xte, yte, sx_te = te[FEATS].values, te.stage.values, te.sex.values
    v2 = []
    for s in seeds:
        A = fit(Xtr, ytr, None, N_EST, s)
        B = fit(Xtr, ytr, inv_freq(ytr) * reweigh(sx_tr, ytr), N_EST, s)
        tf, _ = select_tau_f(Xtr, ytr, sx_tr, s)
        PA, PB = A.predict_proba(Xte), B.predict_proba(Xte)
        v2.append(dict(tau_f=tf,
                       A=pack(yte, PA.argmax(1), PA, sx_te),
                       B=pack(yte, apply_thresholds(PB, sx_te, tf, TAU_M),
                              PB, sx_te)))

    # V3: split reseeded, threshold frozen at the seed-42 value
    tf42 = v1[0]["tau_f"]
    v3 = [full_pipeline(df, s, tau_f=tf42) for s in seeds]

    def agg(runs, label):
        g = {}
        for mdl in ("A", "B"):
            g[mdl] = {}
            for name, fn in (
                    ("acc", lambda r: r[mdl]["acc"] * 100),
                    ("fFNR", lambda r: r[mdl]["female"]["fnr"] * 100),
                    ("EO", lambda r: r[mdl]["eo"]),
                    ("DIR", lambda r: r[mdl]["dir"]),
                    ("late_sens", lambda r: r[mdl]["late_sens"] * 100)):
                v = np.array([fn(r) for r in runs], float)
                g[mdl][name] = dict(mean=round(float(np.nanmean(v)), 3),
                                    sd=round(float(np.nanstd(v, ddof=1)), 3),
                                    vals=[round(float(x), 3) for x in v])
        g["tau_f"] = [round(r["tau_f"], 4) for r in runs]
        out[label] = g

    agg(v1, "V1_full_pipeline_reseeded")
    agg(v2, "V2_test_set_fixed")
    agg(v3, "V3_split_reseeded_threshold_frozen")
    return out, v1[0]


if __name__ == "__main__":
    df = load()
    res = {}

    print("=== reference run, seed 42, tau_f selected by CV on train ===")
    ref = full_pipeline(df, 42)
    print(f"tau_f selected = {ref['tau_f']:.4f}   (dissertation: ~0.02)")
    for m in ("A", "B"):
        r = ref[m]
        print(f"  Model {m}: acc {r['acc']*100:.1f}%  mF1 {r['mf1']:.3f}  "
              f"AUC {r['auc']:.3f}  fFNR {r['female']['fnr']*100:.1f}%  "
              f"mFNR {r['male']['fnr']*100:.1f}%  EO {r['eo']:.3f}  "
              f"DIR {r['dir']:.3f}  sens {r['late_sens']*100:.1f}%")
    res["reference"] = {m: {k: v for k, v in ref[m].items()
                            if k not in ("female", "male")} for m in "AB"}
    res["reference"]["tau_f"] = ref["tau_f"]
    res["reference"]["by_sex"] = {m: {g: ref[m][g] for g in ("female", "male")}
                                  for m in "AB"}

    print("\n=== Q3  CALIBRATION (advanced-stage risk) ===")
    cal = calibration(ref)
    res["calibration"] = cal
    for m in ("A", "B"):
        for g in ("overall", "female", "male"):
            d = cal[m][g]
            print(f"  {m} {g:<8} n={d['n']:<4} prev={d['prevalence']:.3f} "
                  f"meanP={d['mean_pred']:.3f} Brier={d['brier']:.4f} "
                  f"ECE={d['ece']:.4f} CiL={d['calib_in_large']:+.4f}")

    print("\n=== Q2  BOOTSTRAP, fixed test predictions (2000 reps) ===")
    bs = bootstrap(ref, df, B=2000, refit=False)
    res["bootstrap_fixed"] = bs
    for m in ("A", "B"):
        for k, v in bs[m].items():
            print(f"  {m} {k:<5} {v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")

    json.dump(res, open("answers_partial.json", "w"), indent=1, default=str)
    print("\nwrote answers_partial.json")
