# Paper 2 — sex-proxy information decomposition (analysis package)

## Status

**Analysis complete.** The pipeline was run on the cleaned cohort (n = 2,463) and the
manuscript contains real results throughout. Remaining highlighted slots are author,
affiliation, ethics-approval, funding and repository details only.

Preprocessing reproduced the companion study's analytic cohort exactly: Isolation Forest
(contamination 0.015, seed 42) on the five analytes retained 2,472 of 2,510 records,
and removing 9 paediatric records gave n = 2,463 with 1,255 female / 1,208 male, age
49.1 +/- 15.3, and KDIGO counts 604/834/855/103/67 -- every value matching the published table.

### Headline results
| Quantity | Value |
|---|---|
| Staging accuracy, sex-blind | 0.7751 +/- 0.0205 |
| Staging accuracy, sex-aware | 0.9687 +/- 0.0068 |
| Marginal I(creatinine; sex) | 0.0268 nats |
| Conditional I(creatinine; sex \| panel) | 0.0182 nats [0.0115, 0.0260], p = 0.002 |
| Redundancy fraction | 0.319 (so ~68% of the sex signal is unique) |
| Records restaged without the 0.742 multiplier | 634 / 2,463 (25.7%) |

### Headline results — two cohorts x two equations

Staging consequences (equation-dependent):

| | Nigerian CKD-EPI | Nigerian MDRD | NHANES CKD-EPI | NHANES MDRD |
|---|---|---|---|---|
| Records restaged | 529 (21.5%) | 634 (25.7%) | 674 (13.3%) | 1,164 (23.0%) |
| % of women restaged | 42.2% | 50.5% | 25.6% | 44.1% |
| Accuracy sex-blind | 0.7860 | 0.7751 | 0.8730 | 0.8407 |
| Accuracy sex-aware | 0.9639 | 0.9687 | 0.9832 | 0.9823 |
| Cost of omitting sex | 17.8 pts | 19.4 pts | 11.0 pts | 14.2 pts |

Information about MEASURED creatinine (equation-independent):

| Quantity | Nigerian | NHANES |
|---|---|---|
| Conditional I(creatinine; sex \| panel) | 0.0182 [0.0115, 0.0260] | 0.1535 [0.1411, 0.1678] |
| Permutation p | 0.002 | 0.002 |
| Redundancy fraction | 0.319 | 0.092 |
| k-NN estimate at full conditioning | 0.0007 | 0.0061 |

### BROJA partial information decomposition (bits)
| Component | Nigerian | NHANES |
|---|---|---|
| I(creatinine; sex, panel) | 0.1992 | 0.3581 |
| Unique to sex | 0.0379 (19.0%) | 0.2160 (60.3%) |
| Unique to panel | 0.1548 (77.7%) | 0.0923 (25.8%) |
| Redundant | 0.0063 (3.1%) | 0.0378 (10.6%) |
| Synergistic | 0.0003 (0.1%) | 0.0120 (3.3%) |
| **Share of sex information that is unique** | **86%** | **85%** |

`pid.py` was validated on the canonical XOR / copy / unique cases, which it recovers
exactly (CI=1, SI=1, UI1=1 bit respectively). Run `python -c "..."` per the docstring,
or see the validation block in the methods note.

## Contents

| File | What it is |
|---|---|
| `Paper2_manuscript.docx` | Manuscript with real results throughout. | |
| `infotheory.py` | Estimators: KSG MI, KSG conditional MI, Ross mixed MI, Gaussian-copula MI/CMI, bootstrap, permutation test. |
| `validate_estimators.py` | Validation against analytically known ground truth. |
| `validation_output.txt` | Output of the above (already run). |
| `run_analysis.py` | Experiments 1-4 on the cohort -> `results.json`. |
| `pid.py` | BROJA partial information decomposition (validated). |
| `results.json` | Nigerian cohort, MDRD. |
| `results_ckdepi.json` | Nigerian cohort, CKD-EPI 2021. |
| `results_nhanes.json` | NHANES, MDRD. |
| `results_nhanes_ckdepi.json` | NHANES, CKD-EPI 2021. |
| `prepare_cohort.py` | Raw .xlsx -> cleaned cohort CSV (reproduces n = 2,463). |
| `METHODS_NOTE.md` | Why the Gaussian copula is primary and KSG is not. Read this first. |

## Run order

```bash
pip install numpy pandas scipy scikit-learn xgboost

# 1. confirm the estimators behave on this machine
python validate_estimators.py

# 2. run the analysis on the cohort
python run_analysis.py --data /path/to/cohort.csv --out results_ckdepi.json --equation ckdepi
python run_analysis.py --data /path/to/cohort.csv --out results.json        --equation mdrd

# 3. replicate on NHANES
python run_analysis.py --data /path/to/nhanes.csv --creatinine-units mgdl --out results_nhanes_ckdepi.json --equation ckdepi
```

Expected columns: `age, sex, creatinine, urea, sodium, potassium, chloride`.
Creatinine assumed µmol/L; pass `--creatinine-units mgdl` otherwise.
`sex` accepts `0/1`, `M/F` or `Male/Female`.

Runtime is dominated by the bootstrap and permutation loops (1,000 estimator calls,
roughly 3 minutes at n ≈ 2,500). `--skip-exp1` omits model fitting.

## The one thing to read before trusting any output

`METHODS_NOTE.md`. The originally planned k-NN estimator **collapses to zero** at this
conditioning dimension and sample size, which would have produced a false "sex is
redundant" conclusion. The primary estimator was changed to the Gaussian copula on the
strength of that evidence. The dimension-sensitivity table is reported as a result.

## Scripts added after peer review (October 2026)

Creatinine units differ between cohorts: the Nigerian extract is in umol/L, NHANES in mg/dL.
Pass `--creatinine-units` accordingly.

```bash
python prepare_cohort.py --xlsx <nigerian raw>.xlsx        # -> cohort_clean.csv  (n = 2,463)
python prepare_nhanes.py --xlsx <NHANES raw>.xlsx          # -> nhanes_clean.csv  (n = 5,071)

# Paper 2 -- conditional-independence test replacing the global permutation
python conditional_test.py --data cohort_clean.csv --label nigerian --creatinine-units umol
python conditional_test.py --data nhanes_clean.csv --label nhanes  --creatinine-units mgdl

# Paper 2 -- how far the unique-information share depends on the panel representation
python pid_sensitivity.py  --data cohort_clean.csv --label nigerian --creatinine-units umol
python pid_sensitivity.py  --data nhanes_clean.csv --label nhanes  --creatinine-units mgdl

# Paper 3 -- k/sample-size sweep, specificity, non-monotone check, control validation
python diagnostic_study.py                                 # ~20 min; --quick for a fast pass

# Paper 1 -- reproduces the published Model A/B figures, then calibration and seeds
cd paper1_reproduction
python repro_paper1.py cohort_clean.csv                    # recovers n_estimators = 300
python answers.py                                          # calibration + bootstrap scope
python q1_seeds.py                                         # what the five seeds vary
python q1_sensitivity.py                                   # 80-partition stability
```

`diagnostic_study.py` generates its own synthetic cohorts and needs no input data.
