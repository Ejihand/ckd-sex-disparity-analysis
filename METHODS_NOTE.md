# Methods note — estimator selection and the dimensionality failure

**Status:** settled before any patient data was analysed. This note records *why* the
primary estimator is the Gaussian copula and not the k-nearest-neighbour (KSG) estimator
originally proposed. It should be summarised in the manuscript's Methods and is the
answer to the first question a methodological referee will ask.

## The problem we found

The concept note specified the Kraskov–Stögbauer–Grassberger (KSG) k-NN estimator for
`I(creatinine ; sex | age, urea, Na, K, Cl)` — five conditioning variables at n ≈ 2,500.

Before touching patient data, both estimators were run on a synthetic cohort built to the
real cohort's shape (n = 2,463; sex-dependent creatinine medians of 82.7 and 100.5 µmol/L,
matching the study cohort), in which **sex is informative about creatinine by construction**
and in which sodium, potassium and chloride are *pure noise* with respect to creatinine.

Conditioning on noise variables cannot reduce true conditional information. Any estimator
that reports a decline is wrong. Results:

| Conditioning dim. | Gaussian copula (nats) | KSG (nats) |
|---|---|---|
| 1 (age) | 0.0366 | 0.0364 |
| 2 (+ urea) | 0.0338 | 0.0076 |
| 3 (+ sodium) | 0.0338 | 0.0000 |
| 4 (+ potassium) | 0.0338 | 0.0001 |
| 5 (+ chloride) | 0.0340 | 0.0000 |

Marginal `I(creatinine; sex)` on the same data ≈ 0.036 nats.

**KSG collapses to zero once the conditioning set exceeds two variables**, despite the added
variables carrying no information. This is the curse of dimensionality in k-NN density
estimation: neighbourhoods in 7-dimensional joint space at n ≈ 2,500 are too sparse to
resolve the dependence.

The consequence is severe and specific. Had the study been run as originally specified, it
would have reported `I(creatinine ; sex | panel) ≈ 0` and concluded that **sex is redundant
given the routine panel** — a false conclusion that would have looked like a clean positive
finding, supported the fashionable answer, and been very difficult to detect at review.

The Gaussian copula estimator behaves correctly: a small decline when urea (genuinely
creatinine-correlated) enters, then flat as noise is added.

## Decision

- **Primary estimator:** Gaussian copula (Ince et al., 2017). Rank-transform each variable to
  normal scores, then use the closed-form Gaussian expression on the partial correlation.
  It is a **lower bound** on the true information, is robust to monotone nonlinearity, and
  degrades gracefully in conditioning dimension.
- **Secondary:** KSG, reported **only at conditioning dimension ≤ 2**, where validation shows
  it is reliable. It is never used at full conditioning dimension.
- **Reported as a result, not hidden:** the dimension-sensitivity table above is produced for
  the real cohort by `run_analysis.py` and belongs in the supplementary material. It is
  evidence of estimator diligence and pre-empts the obvious referee challenge.

## Known limitation of the chosen estimator

The Gaussian copula captures monotone dependence. Dependence that is strictly non-monotone
(e.g. U-shaped) would be underestimated. Because it returns a lower bound, a **positive and
significant** result is trustworthy; a **null** result must be reported as "no monotone
dependence detected above the estimator's resolution," never as proof of redundancy. This
asymmetry must be stated explicitly in the manuscript's limitations.

## Validation performed

`validate_estimators.py` checks both estimators against analytically known ground truth at
the study's sample size (output in `validation_output.txt`):

- Gaussian MI against `-0.5 ln(1 - ρ²)` — median |relative error| 3.2% across non-null checks.
- Gaussian conditional MI against the partial-correlation form.
- **Conditional independence (the decisive null):** 7 checks, all returning ≈ 0
  (max 0.0213 nats), including three with a *binary* variable mirroring sex.
- Discrete–continuous MI (Ross estimator) against a two-component Gaussian mixture with
  numerically integrated entropy — errors 0.1–8.0%.

Known bias: both estimators are unreliable *in relative terms* when the true value is small
(at ρ = 0.3 the KSG error was −72%, though the absolute error was only 0.034 nats).
Therefore report **absolute values with bootstrap intervals** and establish significance by
**permutation test**, never by the point estimate alone.

## Implication for Experiment 4

The KDIGO stage is a 5-level discrete variable, so the copula transform on it is coarse.
The true-label versus unity-multiplier contrast remains valid as a *difference*, but the
absolute values should be treated cautiously, and the reclassification count (how many
records change stage when the 0.742 multiplier is removed) is the more interpretable and
more robust headline quantity for that experiment.

## Addendum — BROJA decomposition

A bivariate BROJA partial information decomposition was added (`pid.py`). The published
`dit` package would not build in this environment, so the optimisation was implemented
directly: minimise I_q(Y;X1|X2) over joint distributions matching the observed pairwise
marginals, by SQP from several feasible starting points, rejecting any solution that
violates the marginal constraints.

**A first implementation failed validation** and was discarded: normalising inside the
objective made the constraint Jacobian inconsistent, so the optimiser settled on
infeasible points and returned CI = 0.66 for exclusive-or (true value 1 bit) and
UI1 = 0.17 for a purely unique source (true value 1 bit). After the constraint handling
was corrected, all three canonical cases are recovered exactly:

| Case | Expected | Obtained |
|---|---|---|
| XOR | CI = 1 bit | CI = 1.000 |
| Copy (Y = X1 = X2) | SI = 1 bit | SI = 1.000 |
| Unique (Y = X1) | UI1 = 1 bit | UI1 = 1.000 |

No decomposition result should be reported from an implementation that has not passed
these three checks.
