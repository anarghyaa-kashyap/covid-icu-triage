import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parent / "models"   # created by train.py

OUTCOMES = {"death": "Death", "intubation": "Mechanical ventilation", "icu": "ICU admission"}

st.set_page_config(page_title="COVID-19 Risk Predictor", page_icon="🫁", layout="wide")


@st.cache_resource
def load_models():
    return {k: joblib.load(MODEL_DIR / f"{k}.joblib") for k in OUTCOMES if (MODEL_DIR / f"{k}.joblib").exists()}


def predict(art, row):
    X = art["prep"].transform(pd.DataFrame([row])[art["features"]])[:, art["keep_idx"]]
    score = float(art["model"].predict_proba(X)[0, 1])
    pct = float((art["ref_scores"] < score).mean() * 100)          # percentile within Cohort 1
    s = np.clip(score, 1e-6, 1 - 1e-6)
    z = np.log(s / (1 - s))
    prob = float(art["calibrator"].predict_proba([[z]])[0, 1])     # calibrated probability
    return prob, pct


def band(pct):
    if pct < 50: return "Low", "🟢"
    if pct < 80: return "Moderate", "🟡"
    if pct < 95: return "High", "🟠"
    return "Very high", "🔴"


models = load_models()
st.title("🫁 COVID-19 Outcome Risk Predictor")
st.caption("Mini project · Mexico COVID-19 registry (hospitalised, lab-confirmed cases) · "
           "Educational use only, not a medical device.")

if not models:
    st.error(f"No trained models found in {MODEL_DIR}. Run `python train.py` first.")
    st.stop()

tab_pred, tab_perf, tab_about = st.tabs(["Predict", "Model performance", "About"])

with tab_pred:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Patient")
        age = st.slider("Age", 0, 100, 55)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        pregnant = st.checkbox("Pregnant", disabled=(sex == "Male"))
        pneumonia = st.checkbox("Pneumonia diagnosed")
        delay = st.slider("Days from symptom onset to admission", 0, 30, 4)
        st.markdown("**Comorbidities**")
        c1, c2 = st.columns(2)
        flags = {
            "diabetes": c1.checkbox("Diabetes"), "hypertension": c2.checkbox("Hypertension"),
            "obesity": c1.checkbox("Obesity"), "cardiovascular": c2.checkbox("Cardiovascular"),
            "copd": c1.checkbox("COPD"), "asthma": c2.checkbox("Asthma"),
            "chronic_kidney": c1.checkbox("Chronic kidney"), "immunosuppressed": c2.checkbox("Immunosuppressed"),
            "smoker": c1.checkbox("Smoker"), "other_comorbidity": c2.checkbox("Other"),
        }
        go = st.button("Calculate risk", type="primary", use_container_width=True)

    with right:
        st.subheader("Predicted risk")
        if go:
            row = {"age": age, "sex_male": float(sex == "Male"),
                   "pregnant": float(pregnant and sex == "Female"),
                   "pneumonia": float(pneumonia), "symptom_delay_days": delay,
                   **{k: float(v) for k, v in flags.items()}}
            cols = st.columns(len(models))
            table = []
            for col, (key, art) in zip(cols, models.items()):
                prob, pct = predict(art, row)
                label, icon = band(pct)
                with col:
                    st.metric(OUTCOMES[key], f"{prob * 100:.1f}%")
                    st.progress(min(int(pct), 100))
                    st.write(f"{icon} **{label}** · {pct:.0f}th percentile")
                    st.caption(f"Cohort average: {art['baseline_rate'] * 100:.1f}% · model: {art['name']}")
                table.append({"Outcome": OUTCOMES[key], "Probability %": round(prob * 100, 1),
                              "Risk percentile": round(pct), "Band": label})
            st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
            st.info("Risk percentile = how this patient's model score ranks among Cohort-1 patients "
                    "(95th = higher risk than 95% of them).")
        else:
            st.write("Fill in the patient details and press **Calculate risk**.")

with tab_perf:
    mp = MODEL_DIR / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        df = pd.DataFrame([{
            "Outcome": OUTCOMES[r["outcome"]], "Selected model": r["selected"],
            "Cohort-2 AUROC": r["cohort2_auroc"],
            "AUROC 95% CI": f"{r['cohort2_auroc_ci'][0]}–{r['cohort2_auroc_ci'][1]}",
            "Cohort-2 AUPRC": r["cohort2_auprc"],
            "AUPRC 95% CI": f"{r['cohort2_auprc_ci'][0]}–{r['cohort2_auprc_ci'][1]}",
            "Accuracy": r["cohort2_accuracy"], "Balanced acc.": r["cohort2_balanced_accuracy"],
            "Sensitivity": r["cohort2_sensitivity"], "Specificity": r["cohort2_specificity"],
            "Precision": r["cohort2_precision"], "F1": r["cohort2_f1"],
            "Majority-class acc.": r["majority_class_accuracy"],
            "Positives (C2)": f"{r['pos_cohort2']}/{r['n_cohort2']}"} for r in m])
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption("Validated on Cohort 2 = the latest 30% of admissions, never used for training or model selection.")
        img = MODEL_DIR / "cohort2_curves.png"
        if img.exists():
            st.image(str(img), caption="ROC and precision-recall curves on Cohort 2")
        cmp_path = MODEL_DIR / "cohort1_model_comparison.csv"
        if cmp_path.exists():
            with st.expander("All model / feature-processing combinations (Cohort-1 test)"):
                st.dataframe(pd.read_csv(cmp_path), hide_index=True, use_container_width=True)
    else:
        st.warning("metrics.json not found.")

with tab_about:
    st.markdown("""
**Pipeline (adapted from Zhan & Li, CS229):** temporal Cohort 1 / Cohort 2 split by admission date;
raw, SMOTEENN and LASSO+SMOTEENN feature processing; LR, RF, SVM, MLP and LightGBM compared;
best model per outcome selected on a held-out part of Cohort 1 and validated on Cohort 2 with bootstrap CIs.

**Inputs:** age, sex, pregnancy, pneumonia, symptom-to-admission delay, 10 comorbidities.
**Limitations:** the registry has no CT or lab values, so radiomics/lab models from the paper cannot be reproduced;
recent patients may lack final outcomes; associations are not causal.
""")
