# 🎗️ Breast Cancer Risk Prediction System

> An end-to-end machine learning web application for clinical breast cancer risk stratification and early malignancy detection.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Recall](https://img.shields.io/badge/Recall-96.38%25-brightgreen)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.9985-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This project builds a **clinical decision support system** that predicts whether a patient is at **high risk (Malignant)** or **low risk (Benign)** for breast cancer, based on 19 patient health attributes — demographic, lifestyle, physiological, and oncological indicators.

The system trains and compares **3 machine learning models** (Logistic Regression, Random Forest, XGBoost), automatically selects the best one based on **Recall**, and deploys it as an interactive **Streamlit web application** supporting both single-patient and batch cohort screening.

> **Primary Metric:** Recall for Malignant class — because missing a cancer case is clinically far more dangerous than a false alarm.

---

## 🏆 Final Model Performance (XGBoost — Test Set)

| Metric | Score |
|---|---|
| **Recall (Malignant Sensitivity)** | **96.38%** |
| **Precision (PPV)** | **96.13%** |
| **F1-Score** | **96.26%** |
| **ROC-AUC** | **0.9985** |
| **Accuracy** | **98.55%** |
| **Decision Threshold** | **0.50** |

---

## 🗂️ Project Structure

```
breast-cancer-prediction/
│
├── app/
│   └── app.py                        # Streamlit web application
│
├── config/
│   └── config.yaml                   # All project settings & hyperparameters
│
├── data/
│   ├── breast_cancer_prediction.csv  # Raw dataset
│   └── breast_cancer_cleaned.csv     # Cleaned dataset used for training
│
├── models/
│   ├── best_model.pkl                # Trained XGBoost model
│   ├── scaler.pkl                    # Fitted StandardScaler
│   └── encoders.pkl                  # Encoding maps & feature schema
│
├── notebooks/
│   └── 01_eda_and_preprocessing.ipynb  # Exploratory Data Analysis
│
├── outputs/
│   ├── metrics/
│   │   ├── final_metrics.json        # Final test evaluation metrics
│   │   └── model_comparison.json     # All 3 model scores comparison
│   └── plots/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── precision_recall_curve.png
│       └── shap_summary.png
│
├── src/
│   ├── utils.py                      # Config loader, artifact I/O, logging
│   ├── preprocess.py                 # Data cleaning, encoding, scaling, SMOTE
│   ├── train.py                      # Model training & selection pipeline
│   └── evaluate.py                   # Metrics, plots, SHAP explainability
│
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## ⚙️ ML Pipeline

```
Raw Dataset
    ↓
Drop Leakage Columns (Patient_ID, Biopsy_Result, Cancer_Stage)
    ↓
Handle Missing Values (Median imputation / domain defaults)
    ↓
Cap Outliers (BMI clipped to [15, 50])
    ↓
Encode Categories (Binary → 0/1 | Ordinal → 0/1/2 | One-Hot)
    ↓
Train/Test Split (80% / 20%, Stratified)
    ↓
StandardScaler (fit on train only, transform both)
    ↓
SMOTE (applied to training set only — balances class imbalance)
    ↓
Train 3 Models: Logistic Regression | Random Forest | XGBoost
    ↓
GridSearchCV (hyperparameter tuning, CV=3, optimize Recall)
    ↓
Select Best Model by Recall → Save best_model.pkl
    ↓
Final Evaluation on Test Set (Metrics + Confusion Matrix + ROC + SHAP)
    ↓
Deploy as Streamlit Web App
```

---

## 🧪 Models Trained

| Model | Approach | Tuning |
|---|---|---|
| Logistic Regression | Linear baseline, `class_weight=balanced` | None (baseline) |
| Random Forest | 100–200 trees, majority vote | GridSearchCV on `n_estimators`, `max_depth`, `min_samples_split` |
| **XGBoost** ✅ | Sequential boosting, `scale_pos_weight` | GridSearchCV on `max_depth`, `learning_rate`, `n_estimators` |

**XGBoost was automatically selected** as the best model based on the highest Recall score on validation data.

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

**Target:** `Cancer` → `0 = Benign`, `1 = Malignant`

---

## 🌐 Web Application Features

**Single Patient Mode:**
- 19-field clinical intake form across 3 columns
- Instant risk assessment with malignancy probability
- Color-coded risk card (red = HIGH RISK, green = LOW RISK)
- Top 3 contributing risk factors with attribution weights

**Batch Upload Mode:**
- Upload any patient cohort CSV file
- Batch inference across all records
- Cohort summary statistics
- Color-coded results table
- Downloadable annotated predictions CSV

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/breast-cancer-prediction.git
cd breast-cancer-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Dataset
Place `breast_cancer_prediction.csv` into the `data/` folder.

### 5. Run the Full Training Pipeline
```bash
# Step 1: Preprocess data
python -m src.preprocess

# Step 2: Train all models & save best
python -m src.train

# Step 3: Evaluate best model & generate plots
python -m src.evaluate
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
pandas          # Data manipulation
numpy           # Numerical operations
scikit-learn    # ML algorithms, preprocessing, metrics
xgboost         # XGBoost classifier
imbalanced-learn # SMOTE for class imbalance
shap            # Model explainability
matplotlib      # Plotting
seaborn         # Statistical visualization
streamlit       # Web application framework
pyyaml          # YAML config loading
joblib          # Model serialization
jupyter         # EDA notebook
openpyxl        # Excel file support
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Critical Design Decisions

### Data Leakage Prevention
The following columns are **dropped before training** because they represent post-diagnostic information:
- `Patient_ID` — identifier, no predictive value
- `Biopsy_Result` — reveals the diagnosis directly
- `Cancer_Stage` — only known after confirmed diagnosis

### Class Imbalance
The dataset has ~85% Benign and ~15% Malignant patients. We handle this with:
1. **SMOTE** — synthetic oversampling of Malignant class (training only)
2. **`class_weight='balanced'`** in Logistic Regression
3. **`scale_pos_weight`** in XGBoost (ratio of negatives to positives)

### Primary Metric: Recall
We optimize for **Recall** over Accuracy because:
- A False Negative (missed cancer) → patient doesn't get treatment → life-threatening
- A False Positive (false alarm) → patient gets an extra test → much less harmful
- Clinical target: Recall > 90%

### Threshold Calibration
If model Recall < 90%, the `evaluate.py` module automatically searches for a lower decision threshold (0.10 to 0.50) to maximize Recall while maintaining Precision ≥ 70%.

---

## 📊 Output Files

After running the pipeline, the following are generated:

| File | Contents |
|---|---|
| `models/best_model.pkl` | Trained XGBoost model |
| `models/scaler.pkl` | Fitted StandardScaler |
| `models/encoders.pkl` | Encoding maps + feature names |
| `outputs/metrics/final_metrics.json` | Accuracy, Recall, Precision, F1, ROC-AUC |
| `outputs/metrics/model_comparison.json` | All 3 model scores |
| `outputs/plots/confusion_matrix.png` | Annotated confusion matrix |
| `outputs/plots/roc_curve.png` | ROC curve |
| `outputs/plots/precision_recall_curve.png` | PR curve |
| `outputs/plots/shap_summary.png` | SHAP feature importance chart |

---

## ⚕️ Clinical Disclaimer

This application is a **clinical decision support tool only**. It is not a certified medical device and does not constitute a formal medical diagnosis. All risk scores must be interpreted and verified by qualified clinical practitioners and supported by formal oncology imaging and pathology.

---

## 📄 License

This project is licensed under the MIT License.

---

*Built for NiT Hackathon 2026 — Use Case #4: Breast Cancer Prediction*
