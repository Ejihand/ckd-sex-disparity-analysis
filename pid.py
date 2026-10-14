"""
BROJA bivariate partial information decomposition (Bertschinger et al., 2014).

Decomposes the information two sources X1, X2 carry about a target Y into
  UI(X1)  unique to X1
  UI(X2)  unique to X2
  SI      redundant (shared)
  CI      synergistic
with  I(Y; X1,X2) = UI(X1) + UI(X2) + SI + CI.

Definition. Let Q be the set of joint distributions q(y,x1,x2) matching the
observed pairwise marginals p(y,x1) and p(y,x2). Then
    UI(X1) = min_{q in Q} I_q(Y ; X1 | X2)
    UI(X2) = min_{q in Q} I_q(Y ; X2 | X1)
    SI     = I(Y;X1) - UI(X1)
    CI     = I(Y;X1,X2) - min_{q in Q} I_q(Y ; X1,X2)

The optimisation is solved with SLSQP from several starting points; the
marginal constraints are linear, so feasibility is straightforward.

Reference: Bertschinger N, Rauh J, Olbrich E, Jost J, Ay N. Quantifying unique
information. Entropy 2014;16(4):2161-2183.
"""
import numpy as np
from scipy.optimize import minimize

_EPS = 1e-12

__all__ = ["broja_pid", "joint_from_samples", "discretise"]


def discretise(x, bins=4, strategy="quantile"):
    """Map a continuous vector to integer bin labels."""
    x = np.asarray(x, dtype=float)
    if strategy == "quantile":
        edges = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    else:
        edges = np.linspace(x.min(), x.max(), bins + 1)[1:-1]
    return np.digitize(x, edges)


def joint_from_samples(y, x1, x2):
    """Empirical joint p(y,x1,x2) as a 3-D array, from integer-coded samples."""
    y, x1, x2 = [np.asarray(a).astype(int) for a in (y, x1, x2)]
    ny, n1, n2 = y.max() + 1, x1.max() + 1, x2.max() + 1
    p = np.zeros((ny, n1, n2))
    np.add.at(p, (y, x1, x2), 1.0)
    return p / p.sum()


def _mi(p_ab):
    """Mutual information of a 2-D joint, in bits."""
    pa = p_ab.sum(1, keepdims=True)
    pb = p_ab.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = p_ab * np.log2(p_ab / (pa * pb))
    return float(np.nansum(np.where(p_ab > _EPS, t, 0.0)))


def _cmi_y_x1_given_x2(q):
    """I(Y;X1|X2) in bits for a 3-D joint q[y,x1,x2]."""
    p_x2 = q.sum((0, 1))                       # (x2,)
    p_yx2 = q.sum(1)                           # (y,x2)
    p_x1x2 = q.sum(0)                          # (x1,x2)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = q * p_x2[None, None, :]
        den = p_yx2[:, None, :] * p_x1x2[None, :, :]
        t = q * np.log2(num / den)
    return float(np.nansum(np.where(q > _EPS, t, 0.0)))


def _mi_y_x1x2(q):
    """I(Y;(X1,X2)) in bits."""
    ny = q.shape[0]
    return _mi(q.reshape(ny, -1))


def _solve(p, objective, n_restarts=4, seed=0):
    """Minimise `objective(q)` over distributions q matching p's pairwise marginals."""
    ny, n1, n2 = p.shape
    tgt_yx1 = p.sum(2).ravel()
    tgt_yx2 = p.sum(1).ravel()

    # No normalisation inside unpack: the marginal constraints already force
    # sum(q) = 1, and renormalising there makes the constraint Jacobian
    # inconsistent, which lets SLSQP settle on infeasible points.
    def unpack(v):
        return v.reshape(ny, n1, n2)

    cons = [
        {"type": "eq", "fun": lambda v: unpack(v).sum(2).ravel() - tgt_yx1},
        {"type": "eq", "fun": lambda v: unpack(v).sum(1).ravel() - tgt_yx2},
    ]
    bounds = [(0.0, 1.0)] * p.size

    def feasible_start(rng_):
        """Independent coupling of the two target marginals given Y: always feasible."""
        py = p.sum((1, 2))
        q = np.zeros((ny, n1, n2))
        for y in range(ny):
            if py[y] < _EPS:
                continue
            c1 = p.sum(2)[y] / py[y]          # p(x1|y)
            c2 = p.sum(1)[y] / py[y]          # p(x2|y)
            q[y] = py[y] * np.outer(c1, c2)
        if rng_ is not None:
            q = 0.7 * q + 0.3 * rng_.dirichlet(np.ones(q.size)).reshape(q.shape)
            # re-project onto the marginals by one round of scaling
            for _ in range(60):
                m1 = q.sum(2); q *= np.where(m1[:, :, None] > _EPS, (p.sum(2) / np.maximum(m1, _EPS))[:, :, None], 1.0)
                m2 = q.sum(1); q *= np.where(m2[:, None, :] > _EPS, (p.sum(1) / np.maximum(m2, _EPS))[:, None, :], 1.0)
        return q.ravel()

    rng = np.random.default_rng(seed)
    best, best_q = np.inf, None
    for r in range(n_restarts):
        x0 = feasible_start(None if r == 0 else rng)
        try:
            res = minimize(lambda v: objective(unpack(v)), x0, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": 1000, "ftol": 1e-12})
            q = unpack(np.clip(res.x, 0, None))
            # reject solutions that drifted off the constraint set
            viol = max(np.abs(q.sum(2).ravel() - tgt_yx1).max(),
                       np.abs(q.sum(1).ravel() - tgt_yx2).max())
            if viol > 1e-6:
                continue
            val = objective(q)
            if val < best:
                best, best_q = float(val), q
        except Exception:
            continue
    if best_q is None:
        best, best_q = objective(p), p
    return max(best, 0.0), best_q


def broja_pid(p, n_restarts=4, seed=0):
    """
    BROJA decomposition of a 3-D joint p[y,x1,x2]. Returns a dict in bits.
    """
    p = np.asarray(p, dtype=float)
    p = p / p.sum()

    I_total = _mi_y_x1x2(p)
    I_x1 = _mi(p.sum(2))
    I_x2 = _mi(p.sum(1))

    ui1, _ = _solve(p, _cmi_y_x1_given_x2, n_restarts, seed)
    ui2, _ = _solve(np.swapaxes(p, 1, 2), _cmi_y_x1_given_x2, n_restarts, seed)

    si = max(I_x1 - ui1, 0.0)
    si2 = max(I_x2 - ui2, 0.0)
    si = 0.5 * (si + si2)                     # the two routes agree at the optimum
    ci = max(I_total - (ui1 + ui2 + si), 0.0)

    return {"I_total": I_total, "I_x1": I_x1, "I_x2": I_x2,
            "unique_x1": ui1, "unique_x2": ui2,
            "redundant": si, "synergistic": ci,
            "sum_check": ui1 + ui2 + si + ci}
