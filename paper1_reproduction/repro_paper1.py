"""Reproduce Paper 1 / dissertation Model A and Model B.

Spec from dissertation Ch.3:
  features  : age + 5 analytes (sex WITHHELD)
  target    : KDIGO G1..G5 from race-free 4-var MDRD
  split     : 80/20 stratified on stage, random_state=42
  xgb       : max_depth=4, eta=0.08, subsample=0.80, reg_lambda=1.50, multi:softprob
  Model A   : unweighted
  Model B   : inverse class freq x Kamiran-Calders reweighing on (sex, stage)
              + group-conditional threshold on advanced-stage probability
"""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             roc_auc_score, confusion_matrix)
from scipy.stats import fisher_exact
from xgboost import XGBClassifier

CSV = sys.argv[1] if len(sys.argv) > 1 else "cohort_clean.csv"
ANALYTES = ["creatinine", "urea", "sodium", "potassium", "chloride"]
FEATS = ["age"] + ANALYTES


def mdrd(scr_umol, age, sex):
    scr = scr_umol / 88.4
    e = 175 * scr ** -1.154 * age ** -0.203
    return np.where(sex == "female", e * 0.742, e)


def kdigo(egfr):
    return np.select(
        [egfr >= 90, egfr >= 60, egfr >= 30, egfr >= 15],
        [0, 1, 2, 3], default=4)          # 0..4 == G1..G5


def load():
    df = pd.read_csv(CSV)
    df["egfr"] = mdrd(df.creatinine.values, df.age.values, df.sex.values)
    df["stage"] = kdigo(df.egfr.values)
    return df


def reweigh(sex, y):
    """Kamiran & Calders w(a,y) = (N_a * N_y) / (N * N_ay)."""
    n = len(y)
    w = np.ones(n, float)
    for a in np.unique(sex):
        for c in np.unique(y):
            m = (sex == a) & (y == c)
            if m.sum() == 0:
                continue
            w[m] = ((sex == a).sum() * (y == c).sum()) / (n * m.sum())
    return w


def inv_freq(y):
    cnt = np.bincount(y, minlength=5).astype(float)
    cnt[cnt == 0] = 1
    w = len(y) / (5 * cnt)
    return w[y]


def fit(Xtr, ytr, sw, n_est, seed):
    m = XGBClassifier(
        n_estimators=n_est, max_depth=4, learning_rate=0.08,
        subsample=0.80, reg_lambda=1.50, objective="multi:softprob",
        num_class=5, random_state=seed, tree_method="hist",
        eval_metric="mlogloss", verbosity=0)
    m.fit(Xtr, ytr, sample_weight=sw)
    return m


def adv_prob(P):
    """P(advanced) = P(G4) + P(G5)."""
    return P[:, 3] + P[:, 4]


def apply_thresholds(P, sex, tau_f, tau_m):
    """Argmax prediction, overridden to advanced when group threshold met."""
    pred = P.argmax(1)
    pa = adv_prob(P)
    tau = np.where(sex == "female", tau_f, tau_m)
    hit = pa >= tau
    # route to the more likely of G4/G5
    pred = np.where(hit, np.where(P[:, 4] >= P[:, 3], 4, 3), pred)
    return pred


def fairness(ytrue, ypred, sex):
    adv_t = ytrue >= 3
    adv_p = ypred >= 3
    out = {}
    for g in ("female", "male"):
        m = sex == g
        at, ap = adv_t[m], adv_p[m]
        tp = int((at & ap).sum()); fn = int((at & ~ap).sum())
        fp = int((~at & ap).sum()); tn = int((~at & ~ap).sum())
        out[g] = dict(
            n=int(m.sum()), n_adv=tp + fn, tp=tp, fn=fn, fp=fp, tn=tn,
            fnr=fn / (tp + fn) if tp + fn else np.nan,
            tpr=tp / (tp + fn) if tp + fn else np.nan,
            fpr=fp / (fp + tn) if fp + tn else np.nan,
            refer=ap.mean())
    f, m_ = out["female"], out["male"]
    out["eo"] = abs(f["tpr"] - m_["tpr"]) + abs(f["fpr"] - m_["fpr"])
    out["dir"] = f["refer"] / m_["refer"] if m_["refer"] else np.nan
    out["fisher_p"] = fisher_exact([[f["fn"], f["tp"]],
                                    [m_["fn"], m_["tp"]]])[1]
    out["late_sens"] = (adv_t & adv_p).sum() / adv_t.sum()
    return out


def run(df, seed, n_est, tau_f, tau_m):
    tr, te = train_test_split(df, test_size=0.20, random_state=seed,
                              stratify=df.stage)
    Xtr, ytr = tr[FEATS].values, tr.stage.values
    Xte, yte = te[FEATS].values, te.stage.values
    sx_tr, sx_te = tr.sex.values, te.sex.values

    A = fit(Xtr, ytr, None, n_est, seed)
    swB = inv_freq(ytr) * reweigh(sx_tr, ytr)
    B = fit(Xtr, ytr, swB, n_est, seed)

    PA, PB = A.predict_proba(Xte), B.predict_proba(Xte)
    pA = PA.argmax(1)
    pB = apply_thresholds(PB, sx_te, tau_f, tau_m)

    def pack(y, p, P):
        return dict(
            acc=accuracy_score(y, p),
            mf1=f1_score(y, p, average="macro", zero_division=0),
            mrec=recall_score(y, p, average="macro", zero_division=0),
            auc=roc_auc_score(y, P, multi_class="ovr", average="macro"),
            **fairness(y, p, sx_te))

    return dict(n_test=len(te), A=pack(yte, pA, PA), B=pack(yte, pB, PB),
                yte=yte, sx_te=sx_te, PA=PA, PB=PB, pA=pA, pB=pB)


if __name__ == "__main__":
    df = load()
    print(f"n = {len(df)}  stage counts = {np.bincount(df.stage, minlength=5)}")
    print(f"sex = {df.sex.value_counts().to_dict()}\n")
    print("Searching n_estimators for the 74.4% / 0.955 reference run\n")
    print(f"{'n_est':>6} {'acc':>7} {'mF1':>7} {'mRec':>7} {'AUC':>7} "
          f"{'fFNR':>7} {'mFNR':>7} {'EO':>7} {'DIR':>7} {'p':>7} {'sens':>7}")
    for n_est in (50, 100, 150, 200, 300, 400, 500):
        r = run(df, 42, n_est, 0.02, 0.50)["A"]
        print(f"{n_est:>6} {r['acc']*100:>6.1f}% {r['mf1']:>7.3f} "
              f"{r['mrec']:>7.3f} {r['auc']:>7.3f} "
              f"{r['female']['fnr']*100:>6.1f}% {r['male']['fnr']*100:>6.1f}% "
              f"{r['eo']:>7.3f} {r['dir']:>7.3f} {r['fisher_p']:>7.3f} "
              f"{r['late_sens']*100:>6.1f}%")
