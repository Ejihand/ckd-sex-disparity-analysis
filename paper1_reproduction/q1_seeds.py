import json, numpy as np
from answers import full_pipeline, seed_variants, split, select_tau_f, pack
from repro_paper1 import load, fit, reweigh, inv_freq, apply_thresholds, fairness, FEATS

df = load()
out = {}

# --- confirm the published tau_f = 0.02 ----------------------------------
print("=== Model B at the dissertation's stated tau_f = 0.02 ===")
r = full_pipeline(df, 42, tau_f=0.02)
b = r["B"]
print(f"  acc {b['acc']*100:.1f}%  (diss 75.5%)   mF1 {b['mf1']:.3f}   "
      f"AUC {b['auc']:.3f}  (diss 0.955)")
print(f"  fFNR {b['female']['fnr']*100:.1f}%  mFNR {b['male']['fnr']*100:.1f}%  "
      f"EO {b['eo']:.3f} (diss 0.006)  DIR {b['dir']:.3f} (diss 0.609)  "
      f"sens {b['late_sens']*100:.1f}%")
out["tau002"] = {k: v for k, v in b.items() if k not in ("female", "male")}
out["tau002"]["by_sex"] = {g: b[g] for g in ("female", "male")}

# referral counts at tau=0.02, for the price-of-threshold table
yte, sx = r["yte"], r["sx_te"]
for g in ("female", "male"):
    m = sx == g
    d = b[g]
    ppv = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else float("nan")
    print(f"  {g:<7} referred {d['tp']+d['fp']:>3}/{d['n']:<4} "
          f"TP {d['tp']:>2} FP {d['fp']:>3}  PPV {ppv:.3f}")
    out["tau002"]["by_sex"][g]["ppv"] = round(float(ppv), 4)
    out["tau002"]["by_sex"][g]["referred"] = int(d["tp"] + d["fp"])

# --- Q1: what do the five seeds vary? -----------------------------------
print("\n=== Q1  five-seed variants vs published Table 4.10 ===")
print("published: A acc 77.7+-2.0  fFNR 32.5+-1.8  EO 0.33+-0.02  "
      "DIR 0.61+-0.20  sens 83.5+-3.0")
print("           B acc 77.3+-1.7  fFNR  2.0+-2.4  EO 0.04+-0.03  "
      "DIR 0.78+-0.20  sens 98.8+-1.4")
sv, _ = seed_variants(df)
out["seed_variants"] = sv
for label, g in sv.items():
    print(f"\n{label}   tau_f per seed: {g['tau_f']}")
    for m in ("A", "B"):
        s = g[m]
        print(f"  {m}: acc {s['acc']['mean']:.1f}+-{s['acc']['sd']:.1f}  "
              f"fFNR {s['fFNR']['mean']:.1f}+-{s['fFNR']['sd']:.1f}  "
              f"EO {s['EO']['mean']:.3f}+-{s['EO']['sd']:.3f}  "
              f"DIR {s['DIR']['mean']:.2f}+-{s['DIR']['sd']:.2f}  "
              f"sens {s['late_sens']['mean']:.1f}+-{s['late_sens']['sd']:.1f}")

# --- V1 with the threshold frozen at 0.02 (the most likely original) ----
print("\n=== V1b  split reseeded, tau_f frozen at the published 0.02 ===")
runs = [full_pipeline(df, s, tau_f=0.02) for s in (42, 1, 7, 123, 2024)]
g = {}
for m in ("A", "B"):
    g[m] = {}
    for name, fn in (("acc", lambda r: r[m]["acc"] * 100),
                     ("fFNR", lambda r: r[m]["female"]["fnr"] * 100),
                     ("EO", lambda r: r[m]["eo"]),
                     ("DIR", lambda r: r[m]["dir"]),
                     ("late_sens", lambda r: r[m]["late_sens"] * 100)):
        v = np.array([fn(x) for x in runs], float)
        g[m][name] = dict(mean=round(float(np.nanmean(v)), 3),
                          sd=round(float(np.nanstd(v, ddof=1)), 3),
                          vals=[round(float(y), 3) for y in v])
    s = g[m]
    print(f"  {m}: acc {s['acc']['mean']:.1f}+-{s['acc']['sd']:.1f}  "
          f"fFNR {s['fFNR']['mean']:.1f}+-{s['fFNR']['sd']:.1f}  "
          f"EO {s['EO']['mean']:.3f}+-{s['EO']['sd']:.3f}  "
          f"DIR {s['DIR']['mean']:.2f}+-{s['DIR']['sd']:.2f}  "
          f"sens {s['late_sens']['mean']:.1f}+-{s['late_sens']['sd']:.1f}")
    print(f"      per-seed acc {s['acc']['vals']}  fFNR {s['fFNR']['vals']}")
out["V1b_tau_frozen_002"] = g

json.dump(out, open("answers_q1.json", "w"), indent=1, default=str)
print("\nwrote answers_q1.json")
