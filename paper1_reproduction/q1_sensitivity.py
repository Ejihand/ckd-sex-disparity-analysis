"""How much does the reported 5-seed SD depend on WHICH five seeds?

Draws 40 independent sets of five seeds, runs the full pipeline (split
reseeded, tau_f frozen at the published 0.02) and records the mean and SD
of each metric. If the published SDs sit far outside this distribution,
the reported stability is specific to the seed set, not a property of the
pipeline.
"""
import json, numpy as np
from answers import full_pipeline
from repro_paper1 import load

df = load()
rng = np.random.default_rng(7)
NSETS, K = 40, 5

per_seed, cache = {}, {}


def get(s):
    if s not in cache:
        r = full_pipeline(df, int(s), tau_f=0.02)
        cache[s] = dict(
            accA=r["A"]["acc"] * 100, accB=r["B"]["acc"] * 100,
            fnrA=r["A"]["female"]["fnr"] * 100, fnrB=r["B"]["female"]["fnr"] * 100,
            eoA=r["A"]["eo"], eoB=r["B"]["eo"],
            dirA=r["A"]["dir"], dirB=r["B"]["dir"],
            sensA=r["A"]["late_sens"] * 100, sensB=r["B"]["late_sens"] * 100,
            n_adv_f=r["A"]["female"]["n_adv"], n_adv_m=r["A"]["male"]["n_adv"])
    return cache[s]


pool = list(range(1, 81))
print(f"fitting {len(pool)} seeds (2 models each)...")
for s in pool:
    get(s)

keys = ["accA", "fnrA", "eoA", "dirA", "sensA", "accB", "fnrB", "eoB", "dirB", "sensB"]
sets = [rng.choice(pool, K, replace=False) for _ in range(NSETS)]
dist = {k: dict(means=[], sds=[]) for k in keys}
for ss in sets:
    for k in keys:
        v = np.array([get(s)[k] for s in ss], float)
        dist[k]["means"].append(float(np.nanmean(v)))
        dist[k]["sds"].append(float(np.nanstd(v, ddof=1)))

published = dict(accA=(77.7, 2.0), fnrA=(32.5, 1.8), eoA=(0.33, 0.02),
                 dirA=(0.61, 0.20), sensA=(83.5, 3.0),
                 accB=(77.3, 1.7), fnrB=(2.0, 2.4), eoB=(0.04, 0.03),
                 dirB=(0.78, 0.20), sensB=(98.8, 1.4))

print(f"\n{'metric':>6} | {'published':>14} | {'mean of means':>13} | "
      f"{'SD distribution (5-seed sets)':>34} | published SD percentile")
print("-" * 112)
out = {}
for k in keys:
    mu, sd = published[k]
    ms, sds = np.array(dist[k]["means"]), np.array(dist[k]["sds"])
    pct = float((sds < sd).mean() * 100)
    out[k] = dict(published_mean=mu, published_sd=sd,
                  mean_of_means=round(float(ms.mean()), 3),
                  mean_range=[round(float(ms.min()), 2), round(float(ms.max()), 2)],
                  sd_median=round(float(np.median(sds)), 3),
                  sd_range=[round(float(sds.min()), 3), round(float(sds.max()), 3)],
                  published_sd_percentile=round(pct, 1))
    print(f"{k:>6} | {mu:>7.2f} +- {sd:<4.2f} | {ms.mean():>13.2f} | "
          f"median {np.median(sds):>6.3f}  range [{sds.min():.3f}, {sds.max():.3f}] | "
          f"{pct:>5.1f}%")

print("\nadvanced-stage test counts per seed (why the FNR is unstable):")
nf = np.array([get(s)["n_adv_f"] for s in pool])
nm = np.array([get(s)["n_adv_m"] for s in pool])
print(f"  female G4/G5 in test: mean {nf.mean():.1f}  range {nf.min()}-{nf.max()}")
print(f"  male   G4/G5 in test: mean {nm.mean():.1f}  range {nm.min()}-{nm.max()}")
print(f"  one extra missed woman moves the FNR by ~{100/nf.mean():.1f} "
      f"percentage points")
out["n_adv"] = dict(female_mean=round(float(nf.mean()), 2),
                    female_range=[int(nf.min()), int(nf.max())],
                    male_mean=round(float(nm.mean()), 2),
                    male_range=[int(nm.min()), int(nm.max())],
                    fnr_granularity_pp=round(float(100 / nf.mean()), 2))

# direction of effect across all 80 seeds -- the claim that must hold
dA = np.array([get(s)["fnrA"] for s in pool])
dB = np.array([get(s)["fnrB"] for s in pool])
eA = np.array([get(s)["eoA"] for s in pool])
eB = np.array([get(s)["eoB"] for s in pool])
print(f"\nacross all {len(pool)} seeds:")
print(f"  female FNR reduced by mitigation in {(dB < dA).sum()}/{len(pool)} seeds "
      f"(unchanged in {(dB == dA).sum()}, worse in {(dB > dA).sum()})")
print(f"  EO difference reduced in {(eB < eA).sum()}/{len(pool)} seeds")
print(f"  Model A female FNR: mean {dA.mean():.1f}%  SD {dA.std(ddof=1):.1f}  "
      f"range {dA.min():.1f}-{dA.max():.1f}  >0 in {(dA > 0).sum()}/{len(pool)}")
print(f"  Model B female FNR: mean {dB.mean():.1f}%  SD {dB.std(ddof=1):.1f}  "
      f"range {dB.min():.1f}-{dB.max():.1f}")
out["all_seeds"] = dict(
    n_seeds=len(pool),
    fnr_reduced=int((dB < dA).sum()), fnr_equal=int((dB == dA).sum()),
    fnr_worse=int((dB > dA).sum()), eo_reduced=int((eB < eA).sum()),
    A_fnr=dict(mean=round(float(dA.mean()), 2), sd=round(float(dA.std(ddof=1)), 2),
               lo=round(float(dA.min()), 2), hi=round(float(dA.max()), 2),
               n_positive=int((dA > 0).sum())),
    B_fnr=dict(mean=round(float(dB.mean()), 2), sd=round(float(dB.std(ddof=1)), 2),
               lo=round(float(dB.min()), 2), hi=round(float(dB.max()), 2)))
out["per_seed"] = {str(s): {k: round(float(v), 4) for k, v in get(s).items()}
                   for s in pool}

json.dump(out, open("answers_sensitivity.json", "w"), indent=1)
print("\nwrote answers_sensitivity.json")
