"""
Estimator validation against analytically known ground truth (Paper 2, Step 2).

Runs four checks before the estimators are applied to patient data:
  V1  Gaussian MI          -- known: I = -0.5 ln(1 - rho^2)
  V2  Gaussian CMI         -- known: I = -0.5 ln(1 - partial_rho^2)
  V3  Conditional independence -- known: I(X;Y|Z) = 0 when X _||_ Y | Z
  V4  Discrete-continuous MI   -- known: I for a two-component Gaussian mixture

All results are reported at the study's sample size (n = 2463) so that the bias
observed here is the bias that applies to the real analysis.
"""
import numpy as np
from infotheory import (ksg_mi, ksg_cmi, mixed_mi,
                        analytic_gaussian_mi, analytic_gaussian_cmi)

N = 2463          # study sample size
K = 5             # neighbours
SEED = 42
rng = np.random.default_rng(SEED)

rows = []


def report(name, truth, est, n=N):
    err = est - truth
    rel = (err / truth * 100) if abs(truth) > 1e-9 else float('nan')
    rows.append((name, truth, est, err, rel))
    rel_s = f"{rel:+6.1f}%" if np.isfinite(rel) else "   n/a"
    print(f"{name:<46s} truth={truth:7.4f}  est={est:7.4f}  err={err:+7.4f}  {rel_s}")


print("=" * 96)
print(f"ESTIMATOR VALIDATION  (n={N}, k={K}, seed={SEED})   units = nats")
print("=" * 96)

# ---- V1: bivariate Gaussian mutual information ----
print("\nV1. Gaussian MI, I(X;Y) = -0.5 ln(1 - rho^2)")
for rho in (0.0, 0.3, 0.6, 0.9):
    cov = [[1, rho], [rho, 1]]
    xy = rng.multivariate_normal([0, 0], cov, N)
    est = ksg_mi(xy[:, 0], xy[:, 1], k=K, rng=rng)
    report(f"   rho={rho:.1f}", analytic_gaussian_mi(rho), est)

# ---- V2: Gaussian conditional mutual information ----
# Construct X,Y,Z jointly Gaussian; partial correlation of X,Y given Z is known.
print("\nV2. Gaussian CMI, I(X;Y|Z) = -0.5 ln(1 - partial_rho^2)")
for pr in (0.2, 0.5, 0.8):
    # Z independent; X and Y share a component with partial correlation pr given Z
    z = rng.normal(size=N)
    e1 = rng.normal(size=N)
    e2 = rng.normal(size=N)
    common = rng.normal(size=N)
    a = np.sqrt(abs(pr))
    x = 0.7 * z + a * common + np.sqrt(max(1 - a**2, 1e-9)) * e1
    y = 0.7 * z + a * common + np.sqrt(max(1 - a**2, 1e-9)) * e2
    # empirical partial correlation of x,y given z (ground truth for this draw)
    def resid(v, on):
        beta = np.cov(v, on)[0, 1] / np.var(on)
        return v - beta * on
    rx, ry = resid(x, z), resid(y, z)
    pr_emp = np.corrcoef(rx, ry)[0, 1]
    est = ksg_cmi(x, y, z, k=K, rng=rng)
    report(f"   partial_rho={pr_emp:.3f}", analytic_gaussian_cmi(pr_emp), est)

# ---- V3: conditional independence (the critical null) ----
print("\nV3. Conditional independence, I(X;Y|Z) = 0  (X and Y share only Z)")
for trial in range(3):
    z = rng.normal(size=N)
    x = 0.9 * z + 0.5 * rng.normal(size=N)
    y = 0.9 * z + 0.5 * rng.normal(size=N)   # independent of x given z
    est = ksg_cmi(x, y, z, k=K, rng=rng)
    report(f"   trial {trial+1}", 0.0, est)

# ---- V3b: binary conditioning variable (mirrors sex) ----
print("\nV3b. Conditional independence with a BINARY X (mirrors sex)")
for trial in range(3):
    z = rng.normal(size=N)
    s = rng.integers(0, 2, N).astype(float)        # binary, independent of y given z
    y = 0.9 * z + 0.5 * rng.normal(size=N)
    est = ksg_cmi(s, y, z, k=K, rng=rng)
    report(f"   trial {trial+1}", 0.0, est)

# ---- V4: discrete-continuous MI with known truth ----
# C ~ N(mu_d, 1) with mu depending on binary D, equal priors.
# I(C;D) = H(C) - H(C|D); H(C|D) = 0.5*ln(2*pi*e); H(C) computed numerically.
print("\nV4. Discrete-continuous MI, two-component Gaussian mixture (Ross estimator)")
from scipy.integrate import quad
for sep in (0.5, 1.0, 2.0):
    d = rng.integers(0, 2, N)
    c = rng.normal(loc=np.where(d == 1, sep / 2, -sep / 2), scale=1.0)

    def mixpdf(t):
        return 0.5 * (np.exp(-(t - sep/2)**2 / 2) + np.exp(-(t + sep/2)**2 / 2)) / np.sqrt(2*np.pi)
    Hc = -quad(lambda t: mixpdf(t) * np.log(max(mixpdf(t), 1e-300)), -12, 12, limit=400)[0]
    Hc_given_d = 0.5 * np.log(2 * np.pi * np.e)
    truth = Hc - Hc_given_d

    est = mixed_mi(c, d, k=K, rng=rng)
    report(f"   separation={sep:.1f}", truth, est)

# ---- summary ----
print("\n" + "=" * 96)
finite = [r for r in rows if np.isfinite(r[4])]
nulls = [r for r in rows if abs(r[1]) < 1e-9]
print(f"Non-null checks: {len(finite)}   median |relative error| = "
      f"{np.median([abs(r[4]) for r in finite]):.1f}%")
print(f"Null checks (true value 0): {len(nulls)}   max estimate = "
      f"{max(r[2] for r in nulls):.4f} nats")
print("=" * 96)
print("\nInterpretation: the estimator must (a) track known values with modest bias and")
print("(b) return ~0 under conditional independence. Failure of (b) would invalidate any")
print("claim that sex carries unique information, so it is the decisive check.")
