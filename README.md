# Predicting COVID-19 Patient Outcomes at Hospital Admission: Death, Ventilation and ICU Risk Using Machine Learning

**UE24CS352A Machine Learning - Mini Project | Group 2**

| Name | SRN |
|------|-----|
| Bindushree L R | PES1UG24AM901 |
| Anargyaa Kashyap | PES1UG24AM039 |

## Overview
Machine-learning models that predict, from information available at hospital admission, three outcomes for
hospitalised, lab-confirmed COVID-19 patients:

1. **Death**
2. **Mechanical ventilation (intubation)**
3. **ICU admission**

The workflow follows Zhan & Li, *CT-based Patient Triage of COVID-19: Radiomics Prediction of ICU Admission,
Mechanical Ventilation, and Death of Patients* (CS229), adapted to the public Mexican COVID-19 registry
(the paper's CT/lab data is private). A Streamlit app shows a probability and a risk percentile for each outcome.

## Dataset
The dataset is included in this repository as **`covid.zip`** (it contains `covid.csv`, 263,007 patient records).
No unzipping is needed: `train.py` reads the zip directly. We keep patients who are hospitalised and
COVID-positive (35,285 patients).

## Method (short)
- **Features (15):** age, sex, pregnancy, pneumonia, symptom-to-admission delay, 10 comorbidities.
- **Temporal split:** earliest 70% of admissions = Cohort 1 (train/select), latest 30% = Cohort 2 (final unseen test).
  Cohort 1 is further split 70:30 for model selection.
- **Feature processing:** raw, SMOTEENN, LASSO + SMOTEENN.
- **Algorithms:** Logistic Regression, Random Forest, linear SVM (calibrated), MLP, LightGBM (15 combinations per outcome).
- **Selection and validation:** best AUROC on Cohort-1 test; validated on Cohort 2 with 200-resample bootstrap 95% CIs.
- **Metrics:** AUROC, AUPRC, accuracy, balanced accuracy, sensitivity, specificity, precision, F1, Brier score, confusion matrix.
- **Risk output:** calibrated probability (Platt scaling) plus a risk percentile against Cohort-1 patients.
  Each outcome has its own model, so the three probabilities are independent and do not sum to 100%.

## Results on unseen Cohort 2

| Outcome | Selected model | AUROC (95% CI) | Accuracy | Balanced acc. | Sensitivity | Specificity |
|---------|----------------|----------------|----------|---------------|-------------|-------------|
| Death | Raw + linear SVM | 0.643 (0.625-0.661) | 50.6% | 60.4% | 71.9% | 49.0% |
| Ventilation | Raw + LightGBM | 0.719 (0.702-0.737) | 54.7% | 66.7% | 80.0% | 53.4% |
| ICU | SMOTEENN + Random Forest | 0.659 (0.639-0.678) | 43.7% | 62.4% | 83.9% | 40.9% |

Accuracy is measured at a threshold chosen (Youden's J on Cohort-1 test) to favour sensitivity. Always predicting
"no event" would score 93-95% accuracy but detect no patients, so balanced accuracy and sensitivity are more informative.

## Repository structure
```
covid-icu-triage/
├── covid.zip           # dataset (contains covid.csv)
├── train.py            # preprocessing, model comparison, validation, saves models/
├── app.py              # Streamlit app (loads models/)
├── requirements.txt    # Python dependencies
├── README.md
└── models/             # created by train.py (not stored in the repo)
    ├── death.joblib
    ├── intubation.joblib
    ├── icu.joblib
    ├── metrics.json
    ├── cohort1_model_comparison.csv
    └── cohort2_curves.png
```

## Setup and run
Requires Python 3.9 - 3.12.

```bash
# 1. clone the repository and enter it
git clone https://github.com/Bindushree989/covid-icu-triage.git
cd covid-icu-triage

# 2. create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # macOS / Linux

# 3. install dependencies
pip install -r requirements.txt

# 4. train (takes a few minutes; reads covid.zip directly and saves models/)
python train.py covid.zip

# 5. launch the app (opens http://localhost:8501)
streamlit run app.py
```

### Notes
- **Unzip is optional.** If you prefer the CSV, extract it with `unzip covid.zip` (macOS/Linux) or
  *Right-click > Extract All* (Windows), then run `python train.py covid.csv`.
  `python train.py` with no argument looks for `data/covid.csv`.
- `train.py` prints the value counts of `RESULTADO` and `TIPO_PACIENTE` first. If `1` is not COVID-positive in your
  file, change `POSITIVE_RESULT` at the top of `train.py`.
- Lower `MAX_ROWS` in `train.py` (for example 30000) if training is slow.
- Run `train.py` before `app.py`; restart the app after retraining.

## App features
- **Predict:** enter age, sex, pregnancy, pneumonia, days to admission and comorbidities; get probability, risk percentile
  and a Low / Moderate / High / Very high band for each outcome.
- **Model performance:** Cohort-2 metrics table with CIs, ROC/PR curves, and the full model comparison.
- **About:** pipeline summary and limitations.

## Limitations
- No CT or lab data in this registry, so the paper's radiomics and lab models cannot be reproduced.
- Modest discrimination (AUROC 0.64-0.72) and low precision (about 9%).
- Fixed hyperparameters (no cross-validated tuning); missing values and possible right-censoring in registry data.
- Single country and period; **educational use only, not a medical device.**
