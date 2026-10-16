"""
COVID-19 outcome risk prediction (Mexico registry) - local training script.

Run from the project folder:
    python train.py                      # reads data/covid.csv
    python train.py path/to/file.csv     # or give a path

Workflow (follows Zhan & Li, CS229):
  * temporal split by admission date -> Cohort 1 (earliest 70%) / Cohort 2 (latest 30%)
  * Cohort 1 split 70:30 (train / test) for model selection
  * feature processing: raw, SMOTEENN, LASSO + SMOTEENN
  * algorithms: LR, RF, SVM, MLP, LightGBM
  * best combination (AUROC on Cohort-1 test) is validated on Cohort 2 with bootstrap 95% CI
  * reports AUROC, AUPRC, accuracy, balanced accuracy, sensitivity, specificity, precision, F1
Outcomes: death, intubation (mechanical ventilation), ICU admission.
"""
import sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")   # save figures to file, no pop-up window
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve, confusion_matrix, accuracy_score,
                             balanced_accuracy_score, f1_score)
from imblearn.combine import SMOTEENN
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ paths
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "data" / "covid.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ settings
POSITIVE_RESULT = 1     # RESULTADO value meaning COVID-positive (check the printed counts!)
HOSPITALISED = 2        # TIPO_PACIENTE: 1 = outpatient, 2 = hospitalised
MAX_ROWS = 60000        # random cap for speed (SMOTEENN is slow on huge data); None = use all
COHORT1_FRACTION = 0.70
N_BOOT = 200
SEED = 42

COMORBIDITIES = {
    "DIABETES": "diabetes", "EPOC": "copd", "ASMA": "asthma", "INMUSUPR": "immunosuppressed",
    "HIPERTENSION": "hypertension", "OTRA_COM": "other_comorbidity", "CARDIOVASCULAR": "cardiovascular",
    "OBESIDAD": "obesity", "RENAL_CRONICA": "chronic_kidney", "TABAQUISMO": "smoker",
}
FEATURES = (["age", "sex_male", "pregnant", "pneumonia", "symptom_delay_days"]
            + list(COMORBIDITIES.values()))
OUTCOMES = ["death", "intubation", "icu"]
OUTCOME_LABEL = {"death": "Death", "intubation": "Mechanical ventilation", "icu": "ICU admission"}


# ------------------------------------------------------------------ data
def yes_no(s):
    """1 -> 1, 2 -> 0, everything else (97/98/99 = n/a or unknown) -> NaN."""
    return s.map({1: 1.0, 2: 0.0})


def load_and_prepare(path):
    df = pd.read_csv(path, low_memory=False)
    print(f"Raw rows: {len(df):,}")
    print("\nRESULTADO counts:\n", df["RESULTADO"].value_counts().to_string())
    print("\nTIPO_PACIENTE counts:\n", df["TIPO_PACIENTE"].value_counts().to_string())

    df = df[(df["RESULTADO"] == POSITIVE_RESULT) & (df["TIPO_PACIENTE"] == HOSPITALISED)].copy()
    print(f"\nPositive + hospitalised rows: {len(df):,}")
    if len(df) == 0:
        raise SystemExit("No rows left - check POSITIVE_RESULT / HOSPITALISED against the counts above.")

    for c in ["FECHA_INGRESO", "FECHA_SINTOMAS", "FECHA_DEF"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")     # '9999-99-99' -> NaT

    out = pd.DataFrame(index=df.index)
    out["age"] = pd.to_numeric(df["EDAD"], errors="coerce").clip(0, 110)
    out["sex_male"] = (df["SEXO"] == 2).astype(float)
    out["pregnant"] = yes_no(df["EMBARAZO"]).fillna(0)       # 97 = not applicable
    out["pneumonia"] = yes_no(df["NEUMONIA"])
    delay = (df["FECHA_INGRESO"] - df["FECHA_SINTOMAS"]).dt.days
    out["symptom_delay_days"] = delay.where(delay.between(0, 60))
    for src, name in COMORBIDITIES.items():
        out[name] = yes_no(df[src])

    out["death"] = df["FECHA_DEF"].notna().astype(float)
    out["intubation"] = yes_no(df["INTUBADO"])
    out["icu"] = yes_no(df["UCI"])
    out["admit_date"] = df["FECHA_INGRESO"]
    out = out.dropna(subset=["admit_date", "age"])
    if MAX_ROWS and len(out) > MAX_ROWS:
        out = out.sample(MAX_ROWS, random_state=SEED)
        print(f"Randomly sampled {MAX_ROWS:,} rows for speed.")
    return out.sort_values("admit_date").reset_index(drop=True)


# ------------------------------------------------------------------ models
def make_algos():
    return {
        "LR": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=SEED),
        "SVM": CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=5000), cv=3),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=200, random_state=SEED),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                   random_state=SEED, verbose=-1),
    }


def smoteenn(X, y):
    return SMOTEENN(sampling_strategy=0.5, random_state=SEED).fit_resample(X, y)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def run_outcome(data, outcome):
    d = data.dropna(subset=[outcome]).reset_index(drop=True)
    n1 = int(len(d) * COHORT1_FRACTION)
    c1, c2 = d.iloc[:n1], d.iloc[n1:]
    print(f"\n{'=' * 70}\n{OUTCOME_LABEL[outcome]}: n={len(d):,}  prevalence={d[outcome].mean():.3%}")
    print(f"Cohort 1 (to {c1.admit_date.max().date()}): {len(c1):,}   "
          f"Cohort 2 (from {c2.admit_date.min().date()}): {len(c2):,}")
    if c1[outcome].sum() < 30 or c2[outcome].sum() < 10:
        print("Too few positive cases - skipping this outcome.")
        return None

    tr, te = train_test_split(c1, test_size=0.30, stratify=c1[outcome], random_state=SEED)
    ytr, yte, y2 = tr[outcome].values, te[outcome].values, c2[outcome].values

    prep = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler()).fit(tr[FEATURES])
    Xtr, Xte, X2 = (prep.transform(x[FEATURES]) for x in (tr, te, c2))
    n_feat = Xtr.shape[1]
    all_idx = np.arange(n_feat)

    # LASSO feature selection
    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.1).fit(Xtr, ytr)
    keep = np.where(np.abs(lasso.coef_[0]) > 1e-6)[0]
    if len(keep) < 3:
        keep = all_idx
    print("LASSO kept:", [FEATURES[i] for i in keep])

    Xs, ys = smoteenn(Xtr, ytr)
    Xl, yl = smoteenn(Xtr[:, keep], ytr)
    variants = {"raw": (all_idx, Xtr, ytr),
                "SMOTEENN": (all_idx, Xs, ys),
                "LASSO+SMOTEENN": (keep, Xl, yl)}

    rows, best = [], None
    for vname, (idx, Xv, yv) in variants.items():
        for aname, model in make_algos().items():
            model.fit(Xv, yv)
            p = model.predict_proba(Xte[:, idx])[:, 1]
            auc, ap = roc_auc_score(yte, p), average_precision_score(yte, p)
            rows.append({"outcome": outcome, "processing": vname, "algorithm": aname,
                         "AUROC": round(auc, 4), "AUPRC": round(ap, 4)})
            if best is None or auc > best["auc"]:
                best = {"auc": auc, "model": model, "idx": idx, "name": f"{vname} + {aname}", "p_te": p}
        print(f"  finished {vname}")
    table = pd.DataFrame(rows).sort_values("AUROC", ascending=False)
    print("\nCohort-1 test results (top 5):\n", table.head(5).to_string(index=False))
    print("Selected:", best["name"])

    # probability calibration (resampling inflates raw scores) fitted on Cohort-1 test
    calib = LogisticRegression().fit(logit(best["p_te"]).reshape(-1, 1), yte)
    ref = np.sort(best["model"].predict_proba(prep.transform(c1[FEATURES])[:, best["idx"]])[:, 1])

    # validation on Cohort 2
    p2 = best["model"].predict_proba(X2[:, best["idx"]])[:, 1]
    p2_cal = calib.predict_proba(logit(p2).reshape(-1, 1))[:, 1]
    # decision threshold chosen on Cohort-1 test (Youden's J), then applied unchanged to Cohort 2
    fpr_t, tpr_t, thr_t = roc_curve(yte, best["p_te"])
    thr = float(thr_t[np.argmax(tpr_t - fpr_t)])
    pred2 = (p2 >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y2, pred2, labels=[0, 1]).ravel()
    rng = np.random.default_rng(0)
    aucs, aps = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y2), len(y2))
        if 0 < y2[i].sum() < len(i):
            aucs.append(roc_auc_score(y2[i], p2[i]))
            aps.append(average_precision_score(y2[i], p2[i]))
    metrics = {
        "outcome": outcome, "selected": best["name"],
        "n_cohort1": int(len(c1)), "n_cohort2": int(len(c2)),
        "pos_cohort1": int(c1[outcome].sum()), "pos_cohort2": int(y2.sum()),
        "cohort1_test_auroc": round(float(best["auc"]), 4),
        "cohort2_auroc": round(float(roc_auc_score(y2, p2)), 4),
        "cohort2_auroc_ci": [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)],
        "cohort2_auprc": round(float(average_precision_score(y2, p2)), 4),
        "cohort2_auprc_ci": [round(float(np.percentile(aps, 2.5)), 4), round(float(np.percentile(aps, 97.5)), 4)],
        "threshold": round(thr, 4),
        "cohort2_accuracy": round(float(accuracy_score(y2, pred2)), 4),
        "cohort2_balanced_accuracy": round(float(balanced_accuracy_score(y2, pred2)), 4),
        "cohort2_sensitivity": round(float(tp / max(tp + fn, 1)), 4),
        "cohort2_specificity": round(float(tn / max(tn + fp, 1)), 4),
        "cohort2_precision": round(float(tp / max(tp + fp, 1)), 4),
        "cohort2_f1": round(float(f1_score(y2, pred2)), 4),
        "majority_class_accuracy": round(float(1 - y2.mean()), 4),
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        "cohort2_brier_calibrated": round(float(brier_score_loss(y2, p2_cal)), 4),
    }
    print("\nCohort 2 validation:", json.dumps(metrics, indent=2))

    joblib.dump({"outcome": outcome, "features": FEATURES, "prep": prep, "keep_idx": best["idx"],
                 "model": best["model"], "calibrator": calib, "ref_scores": ref,
                 "baseline_rate": float(c1[outcome].mean()), "threshold": thr, "name": best["name"]},
                MODEL_DIR / f"{outcome}.joblib")
    return metrics, table, (y2, p2)


# ------------------------------------------------------------------ main
data = load_and_prepare(CSV_PATH)
all_metrics, tables, curves = [], [], {}
for oc in OUTCOMES:
    res = run_outcome(data, oc)
    if res:
        all_metrics.append(res[0]); tables.append(res[1]); curves[oc] = res[2]

(MODEL_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
pd.concat(tables).to_csv(MODEL_DIR / "cohort1_model_comparison.csv", index=False)

# ROC / PR curves on Cohort 2 (like Fig. 2 of the paper)
fig, axes = plt.subplots(2, len(curves), figsize=(5 * len(curves), 8), squeeze=False)
for j, (oc, (y2, p2)) in enumerate(curves.items()):
    fpr, tpr, _ = roc_curve(y2, p2)
    pr, rc, _ = precision_recall_curve(y2, p2)
    axes[0, j].plot(fpr, tpr); axes[0, j].plot([0, 1], [0, 1], "--", c="gray")
    axes[0, j].set(title=f"{OUTCOME_LABEL[oc]} ROC (AUC={roc_auc_score(y2, p2):.3f})",
                   xlabel="False positive rate", ylabel="True positive rate")
    axes[1, j].plot(rc, pr)
    axes[1, j].set(title=f"{OUTCOME_LABEL[oc]} PR (AP={average_precision_score(y2, p2):.3f})",
                   xlabel="Recall", ylabel="Precision")
plt.tight_layout()
plt.savefig(MODEL_DIR / "cohort2_curves.png", dpi=150)
plt.close()

print("\nDONE. Saved to:", MODEL_DIR)
print(sorted(p.name for p in MODEL_DIR.iterdir()))
