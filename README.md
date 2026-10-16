# COVID-19 Outcome Risk Predictor

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
(the paper's CT/lab data is private). A Streamlit app returns a probability and a risk percentile for each outcome.

## Method (short)
- **Cohort:** COVID-positive (`RESULTADO = 1`) and hospitalised (`TIPO_PACIENTE = 2`) patients: 35,285 of 263,007 records.
- **Features (15):** age, sex, pregnancy, pneumonia, symptom-to-admission delay, 10 comorbidities.
- **Temporal split:** earliest 70% of admissions = Cohort 1 (train/select), latest 30% = Cohort 2 (final unseen test).
  Cohort 1 is further split 70:30 for model selection.
- **Feature processing:** raw, SMOTEENN, LASSO + SMOTEENN.
- **Algorithms:** Logistic Regression, Random Forest, linear SVM (calibrated), MLP, LightGBM (15 combinations per outcome).
- **Selection and validation:** best AUROC on Cohort-1 test; validated on Cohort 2 with 200-resample bootstrap 95% CIs.
- **Metrics:** AUROC, AUPRC, accuracy, balanced accuracy, sensitivity, specificity, precision, F1, Brier score, confusion matrix.

## Results on unseen Cohort 2

| Outcome | Selected model | AUROC (95% CI) | Accuracy | Balanced acc. | Sensitivity | Specificity |
|---------|----------------|----------------|----------|---------------|-------------|-------------|
| Death | Raw + linear SVM | 0.643 (0.625-0.661) | 50.6% | 60.4% | 71.9% | 49.0% |
| Ventilation | Raw + LightGBM | 0.719 (0.702-0.737) | 54.7% | 66.7% | 80.0% | 53.4% |
| ICU | SMOTEENN + Random Forest | 0.659 (0.639-0.678) | 43.7% | 62.4% | 83.9% | 40.9% |

Accuracy is measured at a threshold chosen (Youden's J on Cohort-1 test) to favour sensitivity. Always predicting
"no event" would score 93-95% accuracy but detect no patients, so balanced accuracy and sensitivity are more informative.
Your numbers may differ slightly if you change `MAX_ROWS` or the random seed.

## Project structure
```
covid_project/
├── train.py            # preprocessing, model comparison, validation, saves models/
├── app.py              # Streamlit app (loads models/)
├── requirements.txt
├── README.md
├── covid.zip/
│   └── covid.csv       # NOT in the repo - add your dataset here
└── models/             # created by train.py
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
# 1. create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. put the dataset at data/covid.csv

# 4. train (takes a few minutes; prints metrics and saves models/)
python train.py                  # or: python train.py path/to/file.csv

# 5. launch the app (opens http://localhost:8501)
streamlit run app.py
```

### Notes
- `train.py` prints the value counts of `RESULTADO` and `TIPO_PACIENTE` first. If `1` is not COVID-positive in your
  file, change `POSITIVE_RESULT` at the top of `train.py`.
- Lower `MAX_ROWS` in `train.py` (for example 30000) if training is slow.
- Run `python train.py` before `streamlit run app.py`; restart the app after retraining.

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
