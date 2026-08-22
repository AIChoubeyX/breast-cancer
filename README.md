# 🎗️ Breast Cancer Risk Prediction System

> An end-to-end machine learning pipeline and web application for breast cancer risk stratification, built as a **methodology demonstration on a synthetic dataset**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Recall](https://img.shields.io/badge/Test%20Recall-99.22%25-brightgreen)
![ROC--AUC](https://img.shields.io/badge/Test%20ROC--AUC-0.9982-brightgreen)
![Data](https://img.shields.io/badge/Dataset-Synthetic-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ⚠️ Read This First

**The dataset is synthetic, and its target label is a deterministic rule.** `Cancer = 1`
exactly when **4 or more** of these eight conditions hold:

`Age ≥ 50` · `BMI ≥ 30` · `Tumor_Size_cm ≥ 3.0` · `Family_History = Yes` ·
`Smoking = Yes` · `Genetic_Mutation = Positive` · `Lymph_Node_Involvement = Yes` ·
`Mammogram_Result ≠ Normal`

That rule reproduces the label with **99.72% accuracy** on complete cases. Three
things follow, and they frame everything else in this README:

- **The near-perfect scores measure the dataset, not the model.** Any competent
  learner scores ~99% against a rule with almost no noise.
- **They are not a symptom of leakage.** Once `Biopsy_Result` and `Cancer_Stage` are
  dropped, no remaining feature exceeds 0.68 univariate AUC. The separability is
  spread across eight legitimate risk factors by construction.
- **The feature importances are not clinical findings.** The generator weights its
  eight conditions equally, so their ordering reflects cohort prevalence, not
  medical significance.

**What this project actually demonstrates** is a leak-free ML pipeline: correct
split discipline, resampling confined to training folds, and a decision threshold
chosen before the test set is touched. See [DECISIONS.md](breast-cancer-prediction/DECISIONS.md)
for the evidence behind every choice.

---

## 📌 Project Overview

The system predicts whether a patient is at **high risk (Malignant)** or **low risk
(Benign)**, from 19 patient attributes (20 model features after one-hot encoding)
spanning demographic, lifestyle, physiological, and oncological indicators.

It trains **three model families under two mutually exclusive imbalance
strategies** (six configurations), selects the best on a held-out validation set,
locks a decision threshold, and only then evaluates once on the test set. The
winner is deployed as a Streamlit app supporting single-patient and batch cohort
screening.

> **Primary metric:** Recall for the Malignant class — a missed cancer is far more
> costly than a false alarm. Accuracy is deliberately *not* a selection criterion:
> at 19.3% prevalence, predicting "benign" for everyone scores 80.7%.

---

## 🏆 Final Model Performance

**XGBoost with `scale_pos_weight`** · test set, single evaluation pass, threshold
locked at **0.50** on validation beforehand.

| Metric | Score |
|---|---|
| **Recall (Sensitivity)** | **99.22%** |
| **Precision (PPV)** | **97.96%** |
| **F1-Score** | **98.59%** |
| **ROC-AUC** | **0.9982** |
| **Average Precision** | **0.9966** |
| **Accuracy** | **99.45%** |
| **Specificity** | **99.50%** |
| **NPV** | **99.81%** |

**Confusion matrix** (2000 test patients, 387 malignant):

|  | Predicted Benign | Predicted Malignant |
|---|---:|---:|
| **Actual Benign** | TN = 1605 | FP = 8 |
| **Actual Malignant** | FN = 3 | TP = 384 |

Test recall (0.9922) slightly **exceeds** validation recall (0.9903), which is the
expected signature of an honest evaluation — the test set was not used to tune
anything, so there is no optimistic bias to unwind.

*These figures describe performance against a near-noiseless synthetic rule. Do not
expect them to transfer to real cohorts.*

---

## 🗂️ Project Structure

```
breast-cancer-prediction/
│
├── app/
│   └── app.py                          # Streamlit web application
│
├── config/
│   └── config.yaml                     # Settings, hyperparameter grids, threshold rule
│
├── data/
│   ├── breast_cancer_prediction.csv    # Raw dataset (10,000 rows)
│   └── breast_cancer_cleaned.csv       # Raw minus 3 dropped columns (see note below)
│
├── models/
│   ├── best_pipeline.pkl               # Full fitted pipeline (impute → scale → model)
│   ├── best_model.pkl                  # Bare classifier, for the app
│   ├── scaler.pkl                      # StandardScaler fitted on TRAIN only
│   └── encoders.pkl                    # Encoding maps, learned medians, locked threshold
│
├── notebooks/
│   └── 01_eda_and_preprocessing.ipynb  # Exploratory Data Analysis
│
├── outputs/
│   ├── metrics/
│   │   ├── final_metrics.json          # Test metrics + alignment report
│   │   ├── model_comparison.json       # All 6 configurations
│   │   ├── shap_feature_importance.csv # COMPLETE ranking, all 20 features
│   │   ├── investigations.json         # Leakage & importance evidence
│   │   └── association_screen.csv      # Per-feature association with target
│   └── plots/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── precision_recall_curve.png  # locked operating point marked
│       ├── shap_feature_importance.png
│       └── shap_summary.png            # beeswarm
│
├── src/
│   ├── utils.py                        # Config loader, artifact I/O, logging
│   ├── preprocess.py                   # Cleaning, encoding, splits, pipeline builders
│   ├── train.py                        # 6 experiments, CV, model + threshold selection
│   ├── evaluate.py                     # Test metrics, plots, SHAP + alignment gate
│   └── investigate.py                  # Leakage / importance investigations
│
├── DECISIONS.md                        # Every methodology decision + its evidence
├── verify_methodology.py               # 23 leakage/correctness checks as assertions
├── requirements.txt
└── README.md
```

> **Note on `breast_cancer_cleaned.csv`:** despite the name, it is the raw file minus
> three columns — 0 cells changed, 0 nulls filled, 0 rows removed. All imputation and
> encoding happens in code, per-fold. The filename is misleading and retained only
> for backward compatibility.

---

## ⚙️ ML Pipeline

The split comes **early**, and everything that learns from data comes after it.

```
Raw Dataset (10,000 rows)
    ↓
Drop Patient_ID, Biopsy_Result, Cancer_Stage      ← target proxies + identifier
    ↓
DETERMINISTIC cleaning (safe before split: fixed rules, identical for every row)
  • constant categorical fills ("No", "Moderate")
  • fixed clinical BMI bounds [15, 50]            ← physiology, not percentiles
    ↓
Encode (binary → 0/1 | ordinal → 0/1/2 | one-hot with FIXED categories)
    ↓
╔══════════════════════════════════════════════════════════════════╗
║  THREE-WAY STRATIFIED SPLIT  (random_state=42)                   ║
║  TRAIN 64% (6400)   VALIDATION 16% (1600)   TEST 20% (2000)      ║
╚══════════════════════════════════════════════════════════════════╝
    ↓
        ┌─────────────── imblearn Pipeline, inside each CV fold ───────────────┐
        │  Median imputation  (LEARNED — fitted per fold)                      │
        │  StandardScaler     (LEARNED — fitted per fold, continuous only)     │
        │  SMOTENC            (fit only; skipped at predict time)              │
        │  Classifier                                                          │
        └──────────────────────────────────────────────────────────────────────┘
    ↓
6 configurations = 3 models × 2 imbalance strategies (never stacked)
GridSearchCV — StratifiedKFold(5), scoring = average_precision   ← threshold-free
    ↓
Select best on VALIDATION (recall → F1 → ROC-AUC)
    ↓
Select decision threshold on VALIDATION, lock it into encoders.pkl
    ↓
════════ TEST SET OPENED HERE, FOR THE FIRST AND ONLY TIME ════════
    ↓
Single evaluation pass at the locked threshold + SHAP (alignment asserted)
    ↓
Deploy as Streamlit Web App (reads the same locked threshold)
```

---

## 🧪 Models Trained

Two imbalance strategies, each run against all three model families, as
**alternatives rather than in combination**:

- **A — `class_weight`:** cost-sensitive learning (`class_weight="balanced"`, or
  `scale_pos_weight` for XGBoost). No resampling.
- **B — `smotenc`:** SMOTENC resampling inside the CV folds. No extra weighting.

Ranked by 5-fold CV average precision on the training split:

| Model | A: `class_weight` | B: `smotenc` | Winner |
|---|---:|---:|---|
| **XGBoost** ✅ | **0.9969** ± 0.0030 | 0.9848 ± 0.0044 | A |
| Random Forest | **0.9882** ± 0.0041 | 0.9620 ± 0.0059 | A |
| Logistic Regression | **0.8989** ± 0.0093 | 0.8794 ± 0.0122 | A |

**Cost-sensitive weighting beat resampling in all three pairings**, by margins
several times the CV standard deviation, and trained ~3× faster. Plausible reason:
a threshold rule has sharp decision boundaries, and interpolating synthetic rows
across them manufactures examples the rule would label differently.

**Selected: XGBoost + `scale_pos_weight`**, with
`{learning_rate: 0.1, max_depth: 3, n_estimators: 200}` — validation recall 0.9903,
precision 0.9808, F1 0.9855, ROC-AUC 0.9973.

Logistic Regression could not reach the 0.70 precision floor at any candidate
threshold; the documented max-F1 fallback fired and logged a warning.

---

## 📋 Input Features

| Feature | Type | Description |
|---|---|---|
| Age | Numerical | Patient age in years |
| Gender | Binary | Male / Female |
| BMI | Numerical | Body Mass Index |
| Family_History | Binary | Family history of breast cancer |
| Smoking | Binary | Smoking history |
| Alcohol_Consumption | Binary | Alcohol consumption |
| Physical_Activity | Ordinal | Low / Moderate / High |
| Hormone_Therapy | Binary | Hormone replacement therapy |
| Menopause_Status | Binary | Pre / Post menopause |
| Genetic_Mutation | Binary | BRCA1/BRCA2 mutation status |
| Tumor_Size_cm | Numerical | Tumor or lesion size in cm |
| Lymph_Node_Involvement | Binary | Lymph node involvement |
| Mammogram_Result | Ordinal | Normal / Suspicious / Abnormal |
| Diabetes | Binary | Diabetes diagnosis |
| Exercise_Days_Per_Week | Numerical | Exercise frequency |
| Breastfeeding_History | Nominal (One-Hot) | Yes / No / Not Applicable |
| Annual_Income_USD | Numerical | Annual income |
| Blood_Pressure | Numerical | Systolic BP in mmHg |
| Cholesterol | Numerical | Total cholesterol mg/dL |

**Target:** `Cancer` → `0 = Benign` (80.7%), `1 = Malignant` (19.3%)

**Only 8 of these 19 carry signal** — the eight in the generative rule. The other
eleven are noise by construction, and the trained model finds this on its own: eight
of them receive mean|SHAP| of *exactly* zero. They are retained rather than pruned
because the model discarding them is a more honest demonstration than removing them
by hand.

---

## 🔍 Feature Importance (SHAP)

Complete unmodified ranking — no reordering, no filtering. Full precision in
[`outputs/metrics/shap_feature_importance.csv`](breast-cancer-prediction/outputs/metrics/shap_feature_importance.csv).

| Rank | Feature | mean\|SHAP\| | % of total |
|---:|---|---:|---:|
| 1 | Mammogram_Result | 2.001160 | 14.29 |
| 2 | Family_History | 1.860532 | 13.29 |
| 3 | Smoking | 1.846722 | 13.19 |
| 4 | Age | 1.794174 | 12.81 |
| 5 | Tumor_Size_cm | 1.779964 | 12.71 |
| 6 | Lymph_Node_Involvement | 1.701512 | 12.15 |
| 7 | BMI | 1.650970 | 11.79 |
| 8 | Genetic_Mutation | 1.327843 | 9.48 |
| — | *— 53× cliff —* | | |
| 9 | Cholesterol | 0.024958 | 0.18 |
| 10 | Exercise_Days_Per_Week | 0.010360 | 0.07 |
| 11 | Annual_Income_USD | 0.002167 | 0.02 |
| 12 | Blood_Pressure | 0.000643 | 0.00 |
| 13–20 | Gender, Alcohol_Consumption, Physical_Activity, Hormone_Therapy, Menopause_Status, Diabetes, Breastfeeding_History ×2 | **0.000000** | 0.00 |

**The structure is the result.** The top 8 are exactly the 8 rule features, packed
into a narrow 9.48%–14.29% band; rank 8 → rank 9 drops 53-fold; ranks 13–20 are
exactly zero.

Three things the ranking is often misread as saying:

- **`Smoking` (#3) above `Age` (#4) is real, not a bug.** The gap is 2.9% relative,
  inside the equal-weight band. Feature-name/SHAP-index misalignment was
  specifically checked for and ruled out — `evaluate.py` now *asserts* alignment and
  raises on mismatch. The generator simply weights `Smoking = Yes` and `Age ≥ 50`
  equally. **Smoking's real-world association with breast cancer is weak and
  debated; this is a fact about the generator, not about oncology.**
- **`Genetic_Mutation` at #8 is not "unimportant."** mean|SHAP| blends effect size
  with reach, and this feature has near-top lift (risk ratio 2.93) on the smallest
  footprint (14.5% prevalence, capturing 33% of cancers). Its cohort average is
  diluted; for a mutation-positive patient it is among the strongest signals present.
- **`Tumor_Size_cm` at #5** has the highest univariate AUC of any retained feature,
  but the generator uses only `[≥ 3.0 cm]` — a step function, so a 9 cm tumour
  carries no more information than a 3.1 cm one.

Feature importance describes how *this model* uses *these columns*. **It is not
evidence of biological causation.**

---

## 🌐 Web Application Features

**Single Patient Mode**
- 19-field clinical intake form across 3 columns
- Instant risk assessment with malignancy probability
- Color-coded risk card (red = HIGH RISK, green = LOW RISK)
- Top 3 contributing risk factors with attribution weights

**Batch Upload Mode**
- Upload any patient cohort CSV
- Batch inference across all records with cohort summary statistics
- Color-coded results table + downloadable annotated CSV
- **Hard-fails** if any of the 8 signal features are missing, and warns explicitly
  when optional modelled columns were filled with defaults

The app reads its decision threshold and imputation medians from `encoders.pkl`
rather than hard-coding them, so it cannot silently drift from the trained model.
Verified against the training pipeline across all 2000 test rows: **maximum absolute
difference 0.0** in both transformed features and predicted probabilities.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/breast-cancer-prediction.git
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

Activate it — Windows:
```bash
venv\Scripts\activate
```

macOS / Linux:
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r breast-cancer-prediction/requirements.txt
```

### 4. Add Dataset
Place `breast_cancer_prediction.csv` into `breast-cancer-prediction/data/`.

### 5. Run the Pipeline

All commands run from the `breast-cancer-prediction/` directory, in order.

Verify preprocessing and print the split summary:
```bash
python -m src.preprocess
```

Train 6 configurations, select the model and lock the threshold:
```bash
python -m src.train
```

Evaluate once on the test set and generate plots + SHAP:
```bash
python -m src.evaluate
```

Reproduce the leakage and feature-importance evidence:
```bash
python -m src.investigate
```

Verify the methodology still holds (23 assertions, exits non-zero on failure):
```bash
python verify_methodology.py
```

### 6. Launch the Web Application
```bash
streamlit run app/app.py
```

### 7. (Optional) Run EDA Notebook
```bash
jupyter notebook notebooks/01_eda_and_preprocessing.ipynb
```

---

## 📦 Dependencies

```
pandas           # Data manipulation
numpy            # Numerical operations
scikit-learn     # ML algorithms, preprocessing, metrics
xgboost          # XGBoost classifier
imbalanced-learn # SMOTENC + leak-free Pipeline
shap             # Model explainability
matplotlib       # Plotting
seaborn          # Statistical visualization
streamlit        # Web application framework
scipy            # Chi-square / correlation tests
pyyaml           # YAML config loading
joblib           # Model serialization
jupyter          # EDA notebook
openpyxl         # Excel file support
```

---

## ⚠️ Critical Design Decisions

Summarised here; the evidence for each is in
[DECISIONS.md](breast-cancer-prediction/DECISIONS.md).

### Data leakage prevention

**Dropped as target proxies:**
- `Patient_ID` — identifier, no predictive value
- `Biopsy_Result` — Cramér's V = 1.000, univariate AUC = 1.000, **zero**
  counter-examples, χ² = 8000 = n exactly. This column *is* the label renamed.
- `Cancer_Stage` — `"No Cancer"` covers exactly the negatives. Downstream of
  diagnosis.

**Investigated and retained:** `Lymph_Node_Involvement` was flagged as a possible
proxy on sound clinical reasoning (nodal status is usually established by biopsy or
surgical staging). The data does not support it. The decisive test is the
off-diagonal count — a risk factor has carriers without the disease, a proxy has
none:

| | `Lymph_Node_Involvement` | `Biopsy_Result` (dropped) |
|---|---:|---:|
| Cancer-free patients at highest-risk level | **983 (60.1%)** | **0 (0%)** |
| Share of all cancers captured | 42.2% | 100% |
| Cramér's V | 0.264 | 1.000 |
| χ² (n = 8000) | 556.8 | **8000.0 = n** |

983 cancer-free patients have positive nodal status and it misses 58% of cancers —
neither is compatible with restating the diagnosis. It behaves like
`Genetic_Mutation` (V = 0.251), not like `Biopsy_Result`. Retaining it is also the
consistent choice: `Tumor_Size_cm` has a *stronger* timing objection and a higher
univariate AUC, so excluding one while keeping the other would be incoherent.

### Preprocessing: deterministic vs learned

A transform may run before the split only if its behaviour is fixed in advance.
Anything that computes a statistic *from* the data is fitted inside the training
folds.

| Transform | Class | When |
|---|---|---|
| Constant categorical fills, fixed BMI bounds, encoding maps | deterministic | before split |
| **Median imputation, StandardScaler mean/std** | **learned** | **inside training folds** |

### Class imbalance: alternatives, not a stack

The dataset is **80.7% Benign / 19.3% Malignant**. Applying SMOTE *and*
`class_weight` *and* `scale_pos_weight` together — as an earlier version did —
multiplies three separate minority-upweighting mechanisms by an amount nobody has
computed. They are now compared as mutually exclusive alternatives (see
[Models Trained](#-models-trained)), and cost-sensitive weighting won outright.

Resampling uses **SMOTENC**, not SMOTE: 13 of 20 columns are categorical, and plain
SMOTE would interpolate them into impossible states like `Family_History = 0.63`.
It runs **inside** an `imblearn.Pipeline`, which applies samplers during `fit` and
skips them during `predict`, so held-out folds keep their natural class balance.

### Primary metric: Recall, but not for tuning

Recall is the *clinical* objective — a false negative risks a life, a false positive
risks an extra test. But optimising recall during hyperparameter search is
degenerate: predicting "malignant" for everyone scores 1.0.

So the two concerns are separated. `GridSearchCV` scores on **average precision**
(threshold-free), and the operating point is chosen afterwards, on validation, to
meet the recall target.

### Threshold selection — chosen on validation, never on test

This was the most consequential fix. The earlier `evaluate.py` searched 81 candidate
thresholds **against the test labels** and reported the best. That number was the
best achievable with hindsight, not an estimate of field performance.

The threshold is now chosen in `train.py` on **validation** data over
`{0.50, 0.45, 0.40, 0.35, 0.30}`, under a rule fixed in advance: highest recall
among candidates with precision ≥ 0.70; ties broken on F1, then on the higher
threshold; if none clears the floor, fall back to max F1 **and warn**. It is then
persisted to `encoders.pkl`, and `evaluate.py` **raises** if that key is missing
rather than defaulting to 0.50 — a silent default would quietly reintroduce the
exact bug this separation prevents.

**Selected: 0.50.** Validation recall was flat at 0.9903 from 0.30 to 0.50 while
precision fell 0.9808 → 0.9027, so the rule took the highest-precision member of a
recall-tied set. (Flat recall is itself a fingerprint of the deterministic rule:
predicted probabilities cluster near 0 and 1, so moving the threshold reclassifies
almost nobody.)

### SHAP alignment asserted, not assumed

A feature-name/index misalignment would produce a plausible-looking but entirely
mislabelled ranking, while raising no error. `verify_feature_alignment()` now
compares data columns against `model.feature_names_in_` element-by-element and
**raises** on any mismatch. Result: 20 data features, 20 model features, exact order
match. `ColumnTransformer` is deliberately avoided because it silently reorders
columns, which would break both SHAP labels and SMOTENC's integer indices.

---

## 📊 Output Files

| File | Contents |
|---|---|
| `models/best_pipeline.pkl` | Full fitted pipeline (impute → scale → classifier) |
| `models/best_model.pkl` | Bare classifier, for the app |
| `models/scaler.pkl` | StandardScaler fitted on TRAIN only |
| `models/encoders.pkl` | Encoding maps, learned medians, **locked decision threshold** |
| `outputs/metrics/final_metrics.json` | Test metrics + confusion matrix + alignment report |
| `outputs/metrics/model_comparison.json` | All 6 configurations, CV and validation scores |
| `outputs/metrics/shap_feature_importance.csv` | **Complete** ranking, all 20 features |
| `outputs/metrics/investigations.json` | Leakage, ablation, prevalence-lift evidence |
| `outputs/metrics/association_screen.csv` | Per-feature association with the target |
| `outputs/plots/confusion_matrix.png` | Annotated confusion matrix |
| `outputs/plots/roc_curve.png` | ROC curve |
| `outputs/plots/precision_recall_curve.png` | PR curve, locked operating point marked |
| `outputs/plots/shap_feature_importance.png` | Ranked bar chart, all features |
| `outputs/plots/shap_summary.png` | SHAP beeswarm |

---

## ⚕️ Clinical Disclaimer

**This is not a medical device and must not inform patient care.**

It is a decision-support *prototype* trained on **synthetic data whose label is a
deterministic 8-condition threshold rule**. It has never been validated on real
patients. It is not certified, not a diagnosis, and not a substitute for oncology
imaging, pathology, or clinical judgement.

The reported metrics describe performance against that synthetic rule and **do not
estimate real-world accuracy**. The feature importances reflect how this model uses
these columns; they are **not evidence of biological causation** and must not be
read as clinical findings.

---

## 📄 License

This project is licensed under the MIT License.

---

*Built for NiT Hackathon 2026 — Use Case #4: Breast Cancer Prediction*
