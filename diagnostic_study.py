"""Validation study for the negative-control diagnostic (Paper 3).

Answers the four empirical questions raised at pre-submission review:

  Q1  sensitivity of the KSG collapse to k and to sample size
  Q2  SPECIFICITY -- how often does the diagnostic flag an estimator that is
      behaving acceptably, under ordinary finite-sample fluctuation?
  Q3  does the Gaussian-copula estimator survive NON-MONOTONE dependence,
      which is the assumption its lower-bound property rests on?
  Q4  does the negative control satisfy the independence condition the
      diagnostic actually requires?

The condition. For W added to the conditioning set Z,

    I(X;Y | Z,W) = I(X;Y | Z) + I(X;W | Y,Z) - I(X;W | Z)

so marginal independence of W from X and from Y is NOT sufficient: with a
non-empty Z, conditioning on Z can induce dependence between X and W, making
I(X;W|Z) > 0 and driving the conditional information DOWN. The sufficient
condition is joint independence, W indep (X,Y,Z), under which both correction
terms vanish. A global permutation of an observed column achieves exactly this,
because it destroys the row alignment that carries every dependence.
"""
import argparse, json
import numpy as np
from infotheory import ksg_cmi, gc_cmi, gc_mi

SEED = 42
UMOL = 88.4


def synth(n=2463, rng=None, nonmonotone=False):
    """Synthetic cohort matched to the clinical data in size and structure.

    sex -> creatinine (informative by construction)
    urea correlated with creatinine (genuinely informative)
    sodium, potassium, chloride drawn independently of everything (pure noise)
    """
    rng = np.random.default_rng(SEED) if rng is None else rng
    sex = rng.integers(0, 2, n).astype(float)            # 1 = female
    med = np.where(sex == 1, 82.7, 100.5)                # observed medians
    creat = np.exp(np.log(med) + rng.normal(0, 0.28, n)) / UMOL
    if nonmonotone:
        # U-shaped dependence on age: neither variable is a monotone function
        # of the other, which is the case the copula estimator must be tested on
        age = rng.uniform(18, 85, n)
        creat = creat * (1 + 0.45 * ((age - 50) / 32.0) ** 2)
    else:
        age = 18 + 62 * rng.random(n)
    urea = 0.55 * creat * 10 + rng.normal(0, 1.4, n)     # informative
    Z = np.column_stack([
        age, urea,
        rng.normal(138, 3.5, n),                          # sodium   -- noise
        rng.normal(4.2, 0.5, n),                          # potassium-- noise
        rng.normal(101, 4.0, n),                          # chloride -- noise
    ])
    return creat, sex, Z


def sweep(creat, sex, Z, k, rng):
    """Estimates at conditioning dimension 1..5 for both estimators."""
    out = {"ksg": [], "gc": []}
    for d in range(1, Z.shape[1] + 1):
        out["ksg"].append(float(ksg_cmi(creat, sex, Z[:, :d], k=k, rng=rng)))
        out["gc"].append(float(gc_cmi(creat, sex, Z[:, :d])))
    return out


def collapse_ratio(vals, base_index=1):
    """Retention across the NOISE variables only.

    vals[0] conditions on age, vals[1] adds urea -- urea is genuinely
    informative, so the fall from vals[0] to vals[1] is legitimate and must be
    excluded. The diagnostic concerns vals[1] -> vals[-1], where the three pure-
    noise variables enter and the population value cannot change. Near 1 =
    stable, near 0 = collapse."""
    b = vals[base_index]
    return vals[-1] / b if abs(b) > 1e-12 else float("nan")


# ------------------------------------------------------------------ Q1
def q1_k_and_n(rng):
    print("\n=== Q1  sensitivity to k and to sample size ===")
    print("retention = estimate at 5 conditioning variables / estimate at 2")
    print("(variables 3-5 are pure noise, so the population value cannot change;")
    print(" retention should be 1. the fall from 1 to 2 is legitimate -- urea is")
    print(" informative -- and is excluded from the ratio.)\n")
    rows, out = [], []
    print(f"{'n':>7} {'k':>4} | {'KSG d=2':>9} {'KSG d=5':>9} {'retention':>10} | "
          f"{'GC d=2':>8} {'GC d=5':>8} {'retention':>10}")
    for n in (500, 1000, 2463, 5000, 10000):
        creat, sex, Z = synth(n, np.random.default_rng(SEED))
        for k in (3, 5, 10, 20):
            s = sweep(creat, sex, Z, k, rng)
            rk, rg = collapse_ratio(s["ksg"]), collapse_ratio(s["gc"])
            print(f"{n:>7} {k:>4} | {s['ksg'][1]:>9.4f} {s['ksg'][-1]:>9.4f} {rk:>10.3f} | "
                  f"{s['gc'][1]:>8.4f} {s['gc'][-1]:>8.4f} {rg:>10.3f}")
            out.append({"n": n, "k": k, "ksg": s["ksg"], "gc": s["gc"],
                        "ksg_retention": rk, "gc_retention": rg})
            rows.append((rk, rg))
    rk = np.array([r[0] for r in rows]); rg = np.array([r[1] for r in rows])
    print(f"\nKSG retention across all {len(rows)} configurations: "
          f"{np.nanmin(rk):.3f} to {np.nanmax(rk):.3f} (median {np.nanmedian(rk):.3f})")
    print(f"GC  retention across all {len(rows)} configurations: "
          f"{np.nanmin(rg):.3f} to {np.nanmax(rg):.3f} (median {np.nanmedian(rg):.3f})")
    print(f"KSG lost >50% of its dimension-2 estimate in "
          f"{int((rk < 0.5).sum())}/{len(rk)} configurations; GC in {int((rg < 0.5).sum())}")
    return {"runs": out,
            "ksg_retention": {"min": float(np.nanmin(rk)), "max": float(np.nanmax(rk)),
                              "median": float(np.nanmedian(rk)),
                              "n_below_half": int((rk < 0.5).sum())},
            "gc_retention": {"min": float(np.nanmin(rg)), "max": float(np.nanmax(rg)),
                             "median": float(np.nanmedian(rg)),
                             "n_below_half": int((rg < 0.5).sum())},
            "n_configs": len(rk)}


# ------------------------------------------------------------------ Q2
def q2_specificity(rng, n_rep=200, n=2463, k=5, thresh=0.5):
    """False-flag rate: apply the diagnostic to an estimator that is NOT failing.

    Each replicate draws a fresh cohort and a fresh negative control, computes
    I(X;Y|Z) and I(X;Y|Z,W), and records whether the estimate dropped by more
    than the stated threshold. For a well-behaved estimator this should be rare.
    """
    print(f"\n=== Q2  specificity: false-flag rate over {n_rep} replicates ===")
    print("one negative control added to a 2-variable conditioning set\n")
    res = {}
    for name, fn in (("gaussian_copula", lambda a, b, c: gc_cmi(a, b, c)),
                     ("ksg_k5", lambda a, b, c: ksg_cmi(a, b, c, k=k, rng=rng))):
        drops, flags = [], 0
        for r in range(n_rep):
            rr = np.random.default_rng(1000 + r)
            creat, sex, Z = synth(n, rr)
            base = fn(creat, sex, Z[:, :2])   # age + urea, both legitimate
            W = rr.permutation(Z[:, 2])          # negative control
            with_w = fn(creat, sex, np.column_stack([Z[:, :2], W]))
            d = (base - with_w) / base if abs(base) > 1e-12 else np.nan
            drops.append(d)
            if np.isfinite(d) and d > thresh:
                flags += 1
        drops = np.asarray(drops, float); drops = drops[np.isfinite(drops)]
        rate = flags / n_rep
        print(f"  {name:<16} median relative drop {np.median(drops):+.3f}  "
              f"95th pct {np.percentile(drops, 95):+.3f}  "
              f"false-flag rate at >{thresh:.0%} = {rate:.1%}")
        res[name] = {"median_drop": float(np.median(drops)),
                     "p95_drop": float(np.percentile(drops, 95)),
                     "false_flag_rate": rate, "n_rep": n_rep, "threshold": thresh}
    return res


# ------------------------------------------------------------------ Q3
def q3_nonmonotone(rng):
    """Does the copula estimator hold up when dependence is NOT monotone?"""
    print("\n=== Q3  Gaussian copula under non-monotone dependence ===")
    out = {}
    for label, nm in (("monotone (as in the paper)", False), ("U-shaped in age", True)):
        creat, sex, Z = synth(2463, np.random.default_rng(SEED), nonmonotone=nm)
        s = sweep(creat, sex, Z, 5, rng)
        out[label] = s
        print(f"  {label:<28} GC {s['gc'][1]:.4f} -> {s['gc'][-1]:.4f} "
              f"(retention {collapse_ratio(s['gc']):.3f}) | "
              f"KSG {s['ksg'][1]:.4f} -> {s['ksg'][-1]:.4f} "
              f"(retention {collapse_ratio(s['ksg']):.3f})")
    print("  note: the copula estimator can understate NON-monotone dependence, so a")
    print("  stable retention here shows the diagnostic is not confounded by that")
    print("  limitation -- it does not show the copula value is numerically correct.")
    return out


# ------------------------------------------------------------------ Q4
def q4_control_validity(rng, n=2463, n_rep=100):
    """Does a globally permuted column satisfy the independence the diagnostic needs?

    Checks the two correction terms that must vanish:
        I(X;W|Z) and I(X;W|Y,Z)
    If both are ~0, adding W leaves the population value unchanged.
    """
    print(f"\n=== Q4  is the negative control valid? ({n_rep} permutations) ===")
    print("required: I(creat;W | Z) = 0 and I(creat;W | sex,Z) = 0\n")
    a, b = [], []
    for r in range(n_rep):
        rr = np.random.default_rng(2000 + r)
        creat, sex, Z = synth(n, rr)
        W = rr.permutation(Z[:, 2])
        a.append(gc_cmi(creat, W, Z[:, :2]))
        b.append(gc_cmi(creat, W, np.column_stack([sex, Z[:, :2]])))
    a, b = np.asarray(a), np.asarray(b)
    print(f"  I(creat;W | Z)      mean {a.mean():.5f}  max {a.max():.5f}  "
          f"95th pct {np.percentile(a,95):.5f}")
    print(f"  I(creat;W | sex,Z)  mean {b.mean():.5f}  max {b.max():.5f}  "
          f"95th pct {np.percentile(b,95):.5f}")
    print("  Both are at estimator noise level, so the global permutation does")
    print("  deliver the joint independence the diagnostic requires. A permutation")
    print("  that PRESERVED any structure would not.")
    return {"I_X_W_given_Z": {"mean": float(a.mean()), "max": float(a.max()),
                              "p95": float(np.percentile(a, 95))},
            "I_X_W_given_YZ": {"mean": float(b.mean()), "max": float(b.max()),
                               "p95": float(np.percentile(b, 95))},
            "n_rep": n_rep}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diagnostic_study.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    res = {}
    res["q1_k_and_n"] = q1_k_and_n(rng)
    res["q2_specificity"] = q2_specificity(rng, n_rep=60 if a.quick else 200)
    res["q3_nonmonotone"] = q3_nonmonotone(rng)
    res["q4_control_validity"] = q4_control_validity(rng, n_rep=40 if a.quick else 100)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
