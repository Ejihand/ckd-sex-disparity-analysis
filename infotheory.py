"""
Information-theoretic estimators for the sex-proxy decomposition study (Paper 2).

Implements:
  - ksg_mi            : Kraskov-Stoegbauer-Grassberger mutual information (estimator 1)
  - ksg_cmi           : Frenzel-Pompe / KSG conditional mutual information I(X;Y|Z)
  - mixed_mi          : Ross (2014) estimator for discrete-continuous MI
  - bootstrap_ci      : nonparametric bootstrap confidence intervals
  - permutation_test  : null distribution by shuffling the protected attribute

Design notes
------------
Sex is binary and the analytes are continuous, so estimators that assume all-continuous
inputs are not directly valid. Two routes are provided: the Ross mixed estimator for the
unconditional case, and dithering (tiny continuous jitter on discrete coordinates) for the
conditional case, which is standard practice for k-NN estimators in the presence of ties.
Dither scale is set well below the measurement resolution of the analytes.

References
  Kraskov A, Stoegbauer H, Grassberger P. Estimating mutual information.
      Phys Rev E 2004;69:066138.
  Frenzel S, Pompe B. Partial mutual information for coupling analysis of
      multivariate time series. Phys Rev Lett 2007;99:204101.
  Ross BC. Mutual information between discrete and continuous data sets.
      PLoS ONE 2014;9(2):e87357.
"""

import numpy as np
from scipy.special import digamma
from scipy.spatial import cKDTree

__all__ = ["ksg_mi", "ksg_cmi", "mixed_mi", "bootstrap_ci", "permutation_test",
           "analytic_gaussian_mi", "analytic_gaussian_cmi"]

_EPS = 1e-12


def _as2d(a):
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _dither(a, rng, scale=1e-10):
    """Break ties on discrete coordinates so k-NN distances are well defined."""
    a = _as2d(a).copy()
    return a + rng.normal(0.0, scale, size=a.shape)


def _count_within(data, radii):
    """
    Number of points strictly within `radii` (max-norm), excluding the point itself.

    Uses the vectorised `return_length` form of query_ball_point: building explicit
    neighbour lists is roughly an order of magnitude slower and dominates runtime
    when the estimator is called inside bootstrap/permutation loops.
    """
    tree = cKDTree(data)
    counts = tree.query_ball_point(data, radii - _EPS, p=np.inf,
                                   return_length=True, workers=-1)
    return counts.astype(float) - 1.0


def ksg_mi(x, y, k=5, rng=None):
    """KSG estimator 1 for I(X;Y) with continuous X, Y. Returns nats."""
    rng = np.random.default_rng() if rng is None else rng
    x, y = _dither(x, rng), _dither(y, rng)
    n = x.shape[0]
    xy = np.hstack([x, y])

    tree = cKDTree(xy)
    # distance to the k-th neighbour in the joint space (max-norm)
    eps = tree.query(xy, k=k + 1, p=np.inf, workers=-1)[0][:, k]

    nx = _count_within(x, eps)
    ny = _count_within(y, eps)

    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return max(mi, 0.0)


def ksg_cmi(x, y, z, k=5, rng=None):
    """
    Frenzel-Pompe conditional mutual information I(X;Y|Z) in nats.

    I = psi(k) - < psi(n_xz+1) + psi(n_yz+1) - psi(n_z+1) >
    """
    rng = np.random.default_rng() if rng is None else rng
    x, y, z = _dither(x, rng), _dither(y, rng), _dither(z, rng)
    xyz = np.hstack([x, y, z])

    tree = cKDTree(xyz)
    eps = tree.query(xyz, k=k + 1, p=np.inf, workers=-1)[0][:, k]

    n_xz = _count_within(np.hstack([x, z]), eps)
    n_yz = _count_within(np.hstack([y, z]), eps)
    n_z = _count_within(z, eps)

    cmi = digamma(k) - np.mean(digamma(n_xz + 1) + digamma(n_yz + 1) - digamma(n_z + 1))
    return max(cmi, 0.0)


def mixed_mi(c, d, k=5, rng=None):
    """
    Ross (2014) estimator for I(C;D) with C continuous (n x p) and D discrete (n,).
    Appropriate for e.g. I(creatinine ; sex). Returns nats.
    """
    rng = np.random.default_rng() if rng is None else rng
    c = _dither(c, rng)
    d = np.asarray(d).ravel()
    n = c.shape[0]

    full_tree = cKDTree(c)
    radii = np.empty(n)
    m_i = np.empty(n)
    k_eff = np.empty(n)

    for label in np.unique(d):
        idx = np.flatnonzero(d == label)
        if idx.size <= k:
            # too few members to support k neighbours; fall back to the largest valid k
            kk = max(idx.size - 1, 1)
        else:
            kk = k
        sub = cKDTree(c[idx])
        dist = sub.query(c[idx], k=kk + 1, p=np.inf)[0][:, kk]
        radii[idx] = dist
        k_eff[idx] = kk

    for i in range(n):
        m_i[i] = len(full_tree.query_ball_point(c[i], radii[i] - _EPS, p=np.inf)) - 1

    _, counts = np.unique(d, return_counts=True)
    n_label = np.empty(n)
    for label, cnt in zip(np.unique(d), counts):
        n_label[d == label] = cnt

    mi = digamma(n) - np.mean(digamma(n_label)) + np.mean(digamma(k_eff)) - np.mean(digamma(m_i + 1))
    return max(mi, 0.0)


def bootstrap_ci(fn, arrays, n_boot=500, alpha=0.05, rng=None):
    """Nonparametric bootstrap CI for an estimator `fn(*arrays)`."""
    rng = np.random.default_rng() if rng is None else rng
    n = len(arrays[0])
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = fn(*[np.asarray(a)[idx] for a in arrays])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(stats)), float(lo), float(hi)


def permutation_test(fn, arrays, shuffle_index=1, n_perm=500, rng=None):
    """
    Permutation test. `shuffle_index` selects which array is permuted
    (the protected attribute). Returns (observed, p_value, null_mean).
    """
    rng = np.random.default_rng() if rng is None else rng
    arrays = [np.asarray(a) for a in arrays]
    observed = fn(*arrays)
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = list(arrays)
        perm[shuffle_index] = rng.permutation(perm[shuffle_index])
        null[b] = fn(*perm)
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return float(observed), float(p), float(np.mean(null))


# ---------- analytic ground truth, for estimator validation ----------

def analytic_gaussian_mi(rho):
    """I(X;Y) in nats for a bivariate standard Gaussian with correlation rho."""
    return -0.5 * np.log(1.0 - rho ** 2)


def analytic_gaussian_cmi(partial_rho):
    """I(X;Y|Z) in nats for jointly Gaussian variables with partial correlation."""
    return -0.5 * np.log(1.0 - partial_rho ** 2)


# ---------------------------------------------------------------------------
# Gaussian-copula estimators (Ince et al. 2017).
#
# Rank-transform each variable to normal scores, then use the closed-form
# Gaussian expression. This yields a robust LOWER BOUND on the true information
# and, unlike k-NN estimators, degrades gracefully as the conditioning set grows
# -- which matters here because the routine panel supplies five conditioning
# variables at n ~ 2500.
# ---------------------------------------------------------------------------

def copula_transform(a):
    """Rank-based transform of each column to standard normal scores."""
    from scipy.stats import norm, rankdata
    a = _as2d(a)
    out = np.empty_like(a, dtype=float)
    n = a.shape[0]
    for j in range(a.shape[1]):
        r = rankdata(a[:, j], method="average") / (n + 1.0)
        out[:, j] = norm.ppf(r)
    return out


def gc_cmi(x, y, z):
    """
    Gaussian-copula conditional mutual information I(X;Y|Z) in nats,
    via the partial correlation of normal-scored variables.
    x, y must be single columns; z may be multivariate.
    """
    X = copula_transform(x).ravel()
    Y = copula_transform(y).ravel()
    Z = copula_transform(z)

    # residualise X and Y on Z by least squares, then correlate the residuals
    Zi = np.column_stack([np.ones(len(X)), Z])
    bx, *_ = np.linalg.lstsq(Zi, X, rcond=None)
    by, *_ = np.linalg.lstsq(Zi, Y, rcond=None)
    rx, ry = X - Zi @ bx, Y - Zi @ by

    denom = np.std(rx) * np.std(ry)
    if denom < _EPS:
        return 0.0
    r = float(np.clip(np.mean(rx * ry) / denom, -0.999999, 0.999999))
    return max(-0.5 * np.log(1.0 - r ** 2), 0.0)


def gc_mi(x, y):
    """Gaussian-copula mutual information I(X;Y) in nats."""
    X = copula_transform(x).ravel()
    Y = copula_transform(y).ravel()
    r = float(np.clip(np.corrcoef(X, Y)[0, 1], -0.999999, 0.999999))
    return max(-0.5 * np.log(1.0 - r ** 2), 0.0)
