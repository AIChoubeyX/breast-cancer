# 🎗️ Breast Cancer Prediction — Complete Project Documentation

> **Written in plain, easy-to-understand language for everyone.**

---

## Table of Contents

1. [What Does This Project Do?](#1-what-does-this-project-do)
2. [Project Folder Structure](#2-project-folder-structure)
3. [File-by-File Explanation](#3-file-by-file-explanation)
4. [Complete Pipeline Diagram](#4-complete-pipeline-diagram)
5. [How Each Model Works](#5-how-each-model-works)
6. [Model Performance Comparison](#6-model-performance-comparison)
7. [Why XGBoost Was Chosen](#7-why-xgboost-was-chosen)
8. [Key Terms Glossary (Simple Language)](#8-key-terms-glossary-simple-language)

---

## 1. What Does This Project Do?

This is a **medical AI web application** that predicts whether a patient has a **high risk of breast cancer** (Malignant) or a **low risk** (Benign), based on their health information.

A doctor or healthcare worker can enter a patient's details — like age, BMI, family history, mammogram result, etc. — and the system will immediately give a risk score with a clear verdict.

**Two main ways to use the app:**
- **Single Patient:** Enter one patient's details and get an instant risk assessment.
- **Batch Upload:** Upload a CSV file with hundreds of patients and get risk scores for all of them at once.

The app uses a machine learning model trained on real breast cancer patient data. Three different models were tested and the best one (**XGBoost**) was automatically selected and saved.

---

## 2. Project Folder Structure

```
breast-cancer-prediction/
│
├── app/
│   └── app.py                  ← The web application (what the user sees)
│
├── config/
│   └── config.yaml             ← Settings and configuration file
│
├── data/
│   ├── breast_cancer_prediction.csv   ← Raw original dataset
│   └── breast_cancer_cleaned.csv      ← Cleaned dataset (used for training)
│
├── models/
│   ├── best_model.pkl          ← The trained AI model (XGBoost)
│   ├── scaler.pkl              ← The number normalizer
│   └── encoders.pkl            ← Category-to-number converters
│
├── notebooks/
│   └── 01_eda_and_preprocessing.ipynb  ← Exploratory data analysis notebook
│
├── outputs/
│   ├── metrics/
│   │   ├── final_metrics.json       ← Final model accuracy scores
│   │   └── model_comparison.json   ← Comparison of all 3 models
│   └── plots/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── precision_recall_curve.png
│       └── shap_summary.png
│
├── src/
│   ├── preprocess.py           ← Data cleaning and preparation
│   ├── train.py                ← Model training
│   ├── evaluate.py             ← Model testing and scoring
│   └── utils.py                ← Helper functions
│
├── requirements.txt            ← Python libraries needed
└── config/config.yaml          ← All project settings
```

---

## 3. File-by-File Explanation

---

### 📄 `config/config.yaml` — The Settings File

Think of this as the **remote control** of the project. All important settings are written here so you can change them without touching the actual code.

**What it controls:**
- **Which columns to drop** from the dataset (like `Patient_ID`, `Biopsy_Result`, `Cancer_Stage`) — these are dropped because they would give unfair hints to the model (this is called **data leakage**)
- **Which column is the target** (the answer we want to predict) → `Cancer` (0 = Benign, 1 = Malignant)
- **Which features are numerical** (numbers like Age, BMI)
- **Which features are categorical** (text labels like Yes/No, Male/Female)
- **How to split data** → 80% for training, 20% for testing
- **Model hyperparameters** → settings for Random Forest and XGBoost tuning
- **The minimum Recall score** the model must achieve → 90%

---

### 📄 `data/breast_cancer_prediction.csv` — Raw Dataset

This is the **original dataset** with all patient records exactly as collected. It may contain noise, missing values, and extra columns that need to be cleaned.

### 📄 `data/breast_cancer_cleaned.csv` — Cleaned Dataset

This is the **processed, ready-to-use version** of the dataset. Missing values are handled, and it is the file that is actually fed into the training pipeline.

**Features in the dataset:**

| Feature | Type | What It Means |
|---|---|---|
| Age | Number | Patient's age in years |
| Gender | Category | Male or Female |
| BMI | Number | Body Mass Index (weight vs height ratio) |
| Family_History | Yes/No | Does any close relative have breast cancer? |
| Smoking | Yes/No | Has the patient ever smoked? |
| Alcohol_Consumption | Yes/No | Does the patient drink alcohol? |
| Physical_Activity | Low/Moderate/High | How active is the patient? |
| Hormone_Therapy | Yes/No | Is the patient on hormone replacement therapy? |
| Menopause_Status | Pre/Post | Has the patient gone through menopause? |
| Genetic_Mutation | Yes/No | Does the patient have BRCA1/BRCA2 mutation? |
| Tumor_Size_cm | Number | Size of any detected tumor/lesion in centimeters |
| Lymph_Node_Involvement | Yes/No | Has cancer spread to lymph nodes? |
| Mammogram_Result | Normal/Suspicious/Abnormal | Result of the mammogram scan |
| Diabetes | Yes/No | Does the patient have diabetes? |
| Exercise_Days_Per_Week | Number | How many days per week the patient exercises |
| Breastfeeding_History | Yes/No/Not Applicable | Has the patient breastfed? |
| Annual_Income_USD | Number | Yearly income (socioeconomic indicator) |
| Blood_Pressure | Number | Systolic blood pressure in mmHg |
| Cholesterol | Number | Total cholesterol in mg/dL |
| **Cancer** | **0 or 1** | **Target: 0 = Benign, 1 = Malignant** |

---

### 📄 `src/utils.py` — Helper Functions

This is the **toolbox** of the project. It contains small, reusable functions that are used by all other files.

**Functions inside:**

| Function | What It Does |
|---|---|
| `setup_logging()` | Sets up log messages so you can track what's happening step by step |
| `load_config()` | Reads the `config.yaml` settings file |
| `save_artifact()` | Saves a trained model or scaler to a `.pkl` file on disk |
| `load_artifact()` | Loads a saved `.pkl` file from disk |
| `ensure_dirs()` | Creates output folders if they don't exist yet |

---

### 📄 `src/preprocess.py` — Data Cleaning & Preparation

This is where **raw messy data gets cleaned and transformed** into a format the AI model can understand. Think of it as preparing ingredients before cooking.

**Step-by-step what it does:**

#### Step 1: Handle Missing Values (`handle_missing`)
Some patient records might have blank cells. We fill them:
- **BMI** missing? → Fill with the **median BMI** of all patients
- **Tumor_Size_cm** missing? → Fill with the **median tumor size**
- **Alcohol_Consumption** missing? → Assume **"No"**
- **Physical_Activity** missing? → Assume **"Moderate"**
- **Hormone_Therapy** missing? → Assume **"No"**

#### Step 2: Fix Extreme Values (`handle_outliers`)
Some values are unrealistically extreme:
- **BMI** below 15 or above 50? → Clamp it to the range **[15, 50]** (a BMI of 5 or 120 is clearly an error)

#### Step 3: Convert Text to Numbers (`encode_features`)
AI models only understand numbers, so we convert all text labels:

- **Binary features** (Yes/No): → `No = 0`, `Yes = 1`
  - Example: `Family_History: Yes → 1`
  - Example: `Smoking: No → 0`
- **Ordinal features** (ordered levels): → assigned numbers in order
  - `Physical_Activity: Low=0, Moderate=1, High=2`
  - `Mammogram_Result: Normal=0, Suspicious=1, Abnormal=2`
- **One-Hot Encoding** for `Breastfeeding_History`:
  - Creates new columns like `Breastfeeding_History_Yes`, `Breastfeeding_History_Not Applicable`

#### Step 4: Split Data (`split_data`)
Divides data into two sets:
- **80% Training set** → Model learns from this
- **20% Test set** → Model is evaluated on this (never seen during training)
- **Stratified split** → ensures both sets have the same proportion of cancerous vs non-cancerous patients

#### Step 5: Scale Numbers (`scale_features`)
Numerical features have very different ranges (Age: 20-90, Income: $10,000-$150,000). We **standardize** them so no single feature dominates just because of its large numbers.
- Uses **StandardScaler**: converts each number to "how many standard deviations away from the mean it is"
- Only fitted on training data, then applied to test data (to avoid leakage)
- Saved as `models/scaler.pkl`

#### Step 6: Fix Class Imbalance with SMOTE (`apply_smote`)
If 90% of patients are Benign and only 10% are Malignant, the model will be biased towards always predicting Benign. **SMOTE** fixes this by **synthetically creating new Malignant patient examples** so the training data becomes balanced.
- SMOTE = Synthetic Minority Over-sampling Technique
- Only applied to **training** data, never to test data

#### Full Pipeline (`preprocess_pipeline`)
This function ties all 6 steps above together in the correct order and returns:
- `X_train`, `X_test` → Feature matrices
- `y_train`, `y_test` → Target labels
- `X_train_res`, `y_train_res` → SMOTE-balanced training data

---

### 📄 `src/train.py` — Model Training

This file **trains three different AI models** and picks the best one.

**Step-by-step what it does:**

#### Step 1: Load preprocessed data
Calls the full preprocessing pipeline to get clean, balanced training data.

#### Step 2: Train 3 models

1. **Logistic Regression** (`train_logistic_regression`)
   - Simple baseline model
   - Uses `class_weight='balanced'` to handle imbalance

2. **Random Forest** (`train_random_forest`)
   - More powerful model using many decision trees
   - Uses **GridSearchCV** to automatically find the best settings:
     - Tries different `n_estimators` (100, 200 trees)
     - Tries different `max_depth` (5, 10, 15 levels deep)
     - Tries different `min_samples_split` (2 or 5)
   - Optimizes for **Recall** (catching as many cancer cases as possible)

3. **XGBoost** (`train_xgboost`)
   - The most powerful model, uses boosting (learns from mistakes)
   - Uses `scale_pos_weight` to handle class imbalance
   - Also uses **GridSearchCV** to tune:
     - `max_depth` (3, 5, 7)
     - `learning_rate` (0.01, 0.1)
     - `n_estimators` (100, 200)
   - Optimizes for **Recall**

#### Step 3: Evaluate all models (`evaluate_on_validation`)
Compares all 3 models on the validation data using: Accuracy, Precision, Recall, F1-Score, ROC-AUC

#### Step 4: Select the best model (`select_best_model`)
- Picks the model with the **highest Recall** (most important in cancer detection)
- Gives a clinical warning if Recall < 90%
- Saves the best model as `models/best_model.pkl`

#### Step 5: Save comparison results
Saves all model scores to `outputs/metrics/model_comparison.json`

---

### 📄 `src/evaluate.py` — Model Testing & Visualization

This file **thoroughly tests the final chosen model** and creates visual charts.

**What it does:**

#### 1. Load Test Data (`load_test_data`)
- Loads the 20% test set (patients the model has **never seen before**)
- Applies the same preprocessing — **but does NOT apply SMOTE** (test data must be real, not synthetic)

#### 2. Evaluate Model (`evaluate_model`)
- Calculates all performance metrics on the test set
- If **Recall < 90%**, it automatically tries adjusting the **decision threshold** (from 0.50 down to 0.10) to improve recall while keeping precision above 70%
- Issues a warning if recall is still below the 90% clinical target

#### 3. Generate Visual Charts
- **Confusion Matrix** (`plot_confusion_matrix`): A grid showing how many patients were correctly/incorrectly classified
- **ROC Curve** (`plot_roc_curve`): Shows how well the model separates cancerous from non-cancerous patients
- **Precision-Recall Curve** (`plot_precision_recall_curve`): Shows the trade-off between catching all cancer cases vs accuracy of predictions

#### 4. SHAP Analysis (`shap_analysis`)
- **SHAP** = SHapley Additive exPlanations
- Tells us **why** the model made a particular prediction
- Identifies the **top 5 most important features** globally
- Generates a beeswarm chart saved to `outputs/plots/shap_summary.png`

#### 5. Save Final Metrics
- Saves all metrics to `outputs/metrics/final_metrics.json`

---

### 📄 `app/app.py` — The Web Application

This is the **face of the project** — the interactive web dashboard built with **Streamlit**. This is what users actually see and interact with.

**Sections of the app:**

#### Section 1: Page Configuration & Styling
- Sets up the page title, icon, and layout
- Applies custom CSS (colors, card styles, fonts) for a professional medical look

#### Section 2: Load AI Artifacts (`load_artifacts`)
- Loads the trained model (`best_model.pkl`)
- Loads the scaler (`scaler.pkl`)
- Loads the encoders (`encoders.pkl`)
- Loads performance metrics (`final_metrics.json`)
- **Cached** with `@st.cache_resource` so they load only once and don't reload on every user interaction

#### Section 3: Preprocessing Input (`preprocess_input`)
- Transforms the form data entered by a user into the exact format the model expects
- Same cleaning steps as training: encode, scale, align columns

#### Section 4: Feature Attribution (`get_top_contributing_features`)
- After a prediction, shows the **top 3 factors** that drove the result
- Uses the model's built-in feature importance scores

#### Section 5: Main Application UI (`main`)
**Mode 1: Single Patient Prediction**
- User fills a 3-column form with 19 patient attributes
- Clicks "Run Risk Assessment"
- App shows:
  - A colored banner: RED card for HIGH RISK, GREEN card for LOW RISK
  - Exact malignancy probability percentage
  - A progress bar as a visual risk gauge
  - Metric cards (Malignancy %, Benign %, Triage Status)
  - Top 3 contributing risk factors

**Mode 2: Batch Upload (CSV)**
- User uploads a CSV file with many patients
- App processes all records at once
- Shows cohort summary (total patients, high risk count, low risk count)
- Color-coded results table (red for high risk, green for low risk)
- Download button to export annotated predictions as CSV

#### Section 6: Model Performance Dashboard
- Always visible at the bottom of the app
- Shows the 5 key metrics: Recall, Precision, F1-Score, ROC-AUC, Accuracy
- Medical disclaimer about using this tool as decision support only

---

### 📄 `models/` — Saved AI Artifacts

| File | What It Is |
|---|---|
| `best_model.pkl` | The trained XGBoost model (the "brain") |
| `scaler.pkl` | The StandardScaler fitted on training data |
| `encoders.pkl` | Dictionary of all encoding maps + feature names |

These are saved using **joblib** and loaded by the app at startup.

---

### 📄 `outputs/` — Results & Visualizations

| File | What It Contains |
|---|---|
| `metrics/final_metrics.json` | Accuracy, Precision, Recall, F1, ROC-AUC of best model |
| `metrics/model_comparison.json` | Scores of all 3 models side-by-side |
| `plots/confusion_matrix.png` | Grid showing correct/wrong predictions |
| `plots/roc_curve.png` | ROC curve chart |
| `plots/precision_recall_curve.png` | PR curve chart |
| `plots/shap_summary.png` | Feature importance beeswarm chart |

---

### 📄 `notebooks/01_eda_and_preprocessing.ipynb` — Exploration Notebook

This Jupyter notebook was used for **Exploratory Data Analysis (EDA)** — the initial investigation of the dataset before any model training.

It typically contains:
- Distribution charts of features (histograms, bar charts)
- Correlation heatmaps (which features relate to cancer)
- Missing value analysis
- Class imbalance visualization
- Initial data cleaning experiments

---

### 📄 `requirements.txt` — Python Dependencies

Lists all the Python libraries needed to run this project:

| Library | What It's Used For |
|---|---|
| `pandas` | Loading and manipulating data tables |
| `numpy` | Mathematical operations on arrays |
| `scikit-learn` | Machine learning tools (Logistic Regression, Random Forest, Scaler, GridSearchCV) |
| `xgboost` | The XGBoost algorithm |
| `imbalanced-learn` | SMOTE for handling class imbalance |
| `shap` | Explaining model predictions |
| `matplotlib` | Drawing charts and graphs |
| `seaborn` | Prettier statistical charts |
| `streamlit` | Building the interactive web app |
| `pyyaml` | Reading the config.yaml file |
| `joblib` | Saving and loading model files |
| `jupyter` | Running the exploratory notebook |
| `openpyxl` | Reading/writing Excel files |

---

## 4. Complete Pipeline Diagram

```mermaid
flowchart TD
    A([🗂️ Raw Dataset\nbreast_cancer_prediction.csv]) --> B

    subgraph PREPROCESS ["⚙️ Preprocessing Pipeline — src/preprocess.py"]
        B[Drop Leakage Columns\nPatient_ID, Biopsy_Result, Cancer_Stage] --> C
        C[Handle Missing Values\nMedian for numbers, Mode for categories] --> D
        D[Cap BMI Outliers\nClip to range 15–50] --> E
        E[Encode Categories\nBinary → 0/1, Ordinal → 0/1/2, One-Hot → new columns] --> F
        F[Train / Test Split\n80% Train | 20% Test\nStratified] --> G
        G[Scale Numerical Features\nStandardScaler fitted on Train only] --> H
        H[Apply SMOTE on Training Data\nBalance minority class synthetically]
    end

    H --> I
    H --> I2

    subgraph TRAIN ["🏋️ Model Training — src/train.py"]
        I([Balanced Training Data\nSMOTE-resampled]) --> J & K & L

        J["Model 1\nLogistic Regression\nclass_weight=balanced"]
        K["Model 2\nRandom Forest\nGridSearchCV Tuning\nOptimize Recall"]
        L["Model 3\nXGBoost\nGridSearchCV Tuning\nscale_pos_weight"]

        J & K & L --> M[Evaluate All 3 Models\nAccuracy, Precision, Recall, F1, ROC-AUC]
        M --> N{Select Best Model\nbased on Recall}
        N --> O[💾 Save Best Model\nmodels/best_model.pkl]
    end

    I2([Original Test Set\nNever SMOTE'd]) --> EVAL

    subgraph EVAL ["📊 Evaluation — src/evaluate.py"]
        P[Load Best Model + Test Set] --> Q
        Q[Compute Final Metrics\nAccuracy, Precision, Recall, F1, ROC-AUC] --> R
        R{Recall < 90%?}
        R -- Yes --> S[Tune Decision Threshold\n0.10 to 0.50]
        R -- No --> T
        S --> T[Generate Plots\nConfusion Matrix, ROC Curve,\nPR Curve, SHAP Summary]
        T --> U[💾 Save Metrics & Plots\noutputs/metrics/ & outputs/plots/]
    end

    O --> APP
    U --> APP

    subgraph APP ["🌐 Streamlit Web App — app/app.py"]
        V[Load Artifacts\nModel + Scaler + Encoders + Metrics] --> W

        W{User Mode?}

        W -- Single Patient --> X[Fill 19-field Form\nDemographic + Clinical + Imaging]
        X --> Y[Preprocess Input\nEncode + Scale + Align Features]
        Y --> Z[Model Prediction\npredict_proba]
        Z --> AA[Show Result\nRisk % + HIGH/LOW Card\n+ Top 3 Feature Drivers]

        W -- Batch CSV --> AB[Upload CSV File\nMany Patients]
        AB --> AC[Preprocess All Records]
        AC --> AD[Batch Inference]
        AD --> AE[Show Summary + Color Table\n+ Download CSV]
    end
```

---

## 5. How Each Model Works

### 🔵 Model 1: Logistic Regression

**In simple words:** Imagine drawing a straight line to separate cancer patients from non-cancer patients on a chart. Logistic Regression does exactly this — it finds the best straight line (or flat surface in many dimensions) to divide the two groups.

**How it learns:**
1. Starts with random weights for each feature
2. Makes predictions and checks how wrong it is
3. Adjusts the weights to reduce the error
4. Repeats until it can't improve anymore

**Why it's used here:**
- Used as a **baseline** — a simple reference model
- If even a simple model performs well, more complex models will do even better
- `class_weight='balanced'` is set so it doesn't ignore the minority (cancer) class

**Limitation:** Can only draw straight boundaries. If the relationship between features and cancer is curved/complex, it may not capture it well.

---

### 🟠 Model 2: Random Forest

**In simple words:** Imagine asking 100-200 different doctors (decision trees) for their opinion on a patient. Each doctor was trained on slightly different data and asks different questions. The final answer is the **majority vote** of all doctors.

**How it learns:**
1. Randomly picks a subset of training data (with replacement) — this is called **bootstrapping**
2. Grows a Decision Tree on each subset (but only uses a random selection of features at each split)
3. Each tree independently decides: Benign or Malignant?
4. Final prediction = **majority vote** of all trees

**Key settings tuned with GridSearchCV:**
- `n_estimators` = number of trees (100 or 200)
- `max_depth` = how deep each tree can grow (5, 10, or 15 levels)
- `min_samples_split` = minimum patients needed to split a node (2 or 5)

**Why it's better than one tree:**
- One tree overfits (memorizes training data)
- Many different trees, each seeing slightly different data, cancel each other's errors out → more reliable predictions

---

### 🟢 Model 3: XGBoost (eXtreme Gradient Boosting)

**In simple words:** Instead of building many trees independently (like Random Forest), XGBoost builds trees **one at a time, each one fixing the mistakes of the previous one.**

Think of it like this: Tree 1 makes predictions. The second tree focuses on the patients Tree 1 got wrong. Tree 3 focuses on what Tree 2 still got wrong. This continues until the errors become very small.

**How it learns:**
1. Starts with a simple initial prediction (e.g., predict everyone is Benign)
2. Calculates how wrong it was (the **residuals** / errors)
3. Builds a new tree to predict these errors
4. Adds a fraction of this tree's predictions to the original prediction (controlled by `learning_rate`)
5. Repeats steps 2-4 for `n_estimators` rounds

**Key settings tuned with GridSearchCV:**
- `max_depth` = tree depth (3, 5, or 7)
- `learning_rate` = how much each tree contributes (0.01 or 0.1 — lower = more conservative but often better)
- `n_estimators` = number of boosting rounds (100 or 200)
- `scale_pos_weight` = automatically computed ratio of negatives to positives, to handle class imbalance

**Why XGBoost is powerful:**
- Learns from mistakes sequentially
- Regularization built-in (prevents overfitting)
- Very fast due to hardware optimization
- Consistently tops machine learning competitions

---

## 6. Model Performance Comparison

All three models are evaluated on the **same 20% test set** that was **never seen during training**. The primary metric for selection is **Recall** — because in cancer detection, missing a real cancer case (False Negative) is far more dangerous than a false alarm.

### Final Metrics of the Selected Model (XGBoost)

| Metric | Score | What It Means |
|---|---|---|
| **Accuracy** | **98.55%** | Out of all patients, 98.55% were correctly classified |
| **Precision** | **96.13%** | Of all patients flagged as HIGH RISK, 96.13% actually had cancer |
| **Recall (Sensitivity)** | **96.38%** | Of all actual cancer patients, 96.38% were correctly caught |
| **F1-Score** | **96.26%** | Balanced score combining Precision and Recall |
| **ROC-AUC** | **0.9985** | Nearly perfect separation between cancer and non-cancer patients |
| **Decision Threshold** | **0.50** | If malignancy probability ≥ 50%, patient is flagged HIGH RISK |

### What These Numbers Mean in Practice

Imagine 1,000 patients, of which 100 actually have cancer:

| Model Decision | Count |
|---|---|
| Correctly identified cancer patients (True Positives) | ~96 out of 100 |
| Missed cancer patients — False Negatives (dangerous!) | ~4 out of 100 |
| Correctly cleared non-cancer patients (True Negatives) | ~880 out of 900 |
| False alarms — flagged as cancer but actually benign | ~20 out of 900 |

---

## 7. Why XGBoost Was Chosen

The model selection is **automatic** — the code compares all three models and selects the one with the **highest Recall** on the validation set.

**XGBoost wins because:**

1. **Highest Recall**: It catches the most actual cancer cases. Missing a cancer case can cost a life, so this is the top priority.

2. **Best ROC-AUC (0.9985)**: Nearly perfect at separating cancerous from non-cancerous patients — barely any overlap between the two classes.

3. **Handles Class Imbalance**: The `scale_pos_weight` parameter automatically adjusts for the fact that there are more non-cancer patients than cancer patients in the dataset. This prevents the model from being biased.

4. **Learns from Mistakes (Boosting)**: Each new tree corrects the errors of the previous one. This makes XGBoost extremely accurate on complex, real-world medical data.

5. **GridSearchCV Tuning**: The best combination of depth, learning rate, and number of trees was automatically found, making it optimally configured.

6. **Clinical Safety**: The code includes a safety check — if the best model's Recall is below 90%, a **warning is issued** to the user suggesting further tuning. This ensures no unsafe model gets deployed.

---

## 8. Key Terms Glossary (Simple Language)

| Term | Simple Explanation |
|---|---|
| **Malignant** | Cancerous — the tumor is harmful and can spread |
| **Benign** | Non-cancerous — the growth is harmless |
| **Recall (Sensitivity)** | Out of all real cancer patients, how many did we catch? High recall = fewer missed cancers. This is the most critical metric here. |
| **Precision (PPV)** | Out of all patients we flagged as HIGH RISK, how many actually had cancer? High precision = fewer false alarms. |
| **Accuracy** | Of ALL patients (cancer + non-cancer), how many did we classify correctly? |
| **F1-Score** | A single number that balances both Precision and Recall. Useful when both matter. |
| **ROC-AUC** | A score from 0 to 1. How well the model separates cancerous from non-cancerous patients overall. 1.0 = perfect, 0.5 = random guessing. |
| **False Negative** | The model says "Low Risk" but the patient actually has cancer. This is the most dangerous error — the model missed a real cancer case. |
| **False Positive** | The model says "HIGH RISK" but the patient doesn't have cancer. A false alarm — stressful but not as dangerous as missing cancer. |
| **True Positive** | The model correctly identified a cancer patient as HIGH RISK. |
| **True Negative** | The model correctly identified a healthy patient as LOW RISK. |
| **Decision Threshold** | By default, if malignancy probability ≥ 50%, the patient is flagged HIGH RISK. This threshold can be lowered (e.g., to 30%) to catch more cancer cases at the cost of more false alarms. |
| **SMOTE** | Synthetic Minority Over-sampling Technique. Creates artificial cancer patient examples to balance the training dataset (because real-world data usually has far fewer cancer cases than healthy cases). |
| **Overfitting** | When a model memorizes the training data so well that it fails on new patients it hasn't seen before. |
| **GridSearchCV** | A method that automatically tries all combinations of model settings and picks the one that performs best. Like trying every possible recipe and keeping the tastiest one. |
| **StandardScaler** | Converts all numbers to the same scale (mean=0, std=1). Prevents features with large numbers (like income: $100,000) from dominating features with small numbers (like age: 45). |
| **Feature Importance** | Which patient attributes (features) influenced the model's decision the most? E.g., Mammogram Result and Genetic Mutation tend to be the most important. |
| **SHAP Values** | A method to explain WHY the model made a specific prediction for a specific patient. Tells you which features pushed the prediction towards cancer or away from it. |
| **Confusion Matrix** | A 2×2 grid showing: True Positives, False Positives, True Negatives, and False Negatives. |
| **ROC Curve** | A graph showing how well the model works at different thresholds. A curve that hugs the top-left corner is better. |
| **Precision-Recall Curve** | A graph showing the trade-off between catching all cancers (recall) and being accurate when you say cancer (precision). |
| **Cross-Validation (CV=3)** | The training data is split into 3 parts. The model trains on 2 parts and is tested on the 3rd, rotating 3 times. Average score is used. This gives a reliable estimate of real-world performance. |
| **Data Leakage** | When information that shouldn't be available at prediction time accidentally gets used during training. Example: `Biopsy_Result` tells us if a patient has cancer — using it to train would give unfairly perfect results. The model must only use information a doctor has BEFORE confirming a diagnosis. |
| **One-Hot Encoding** | Converts a category with multiple values into separate Yes/No columns. Example: `Breastfeeding_History` (Yes/No/Not Applicable) becomes two new columns. |
| **Ordinal Encoding** | Converts ordered categories to numbers: Low=0, Moderate=1, High=2. The order is preserved. |
| **Binary Encoding** | Converts Yes/No labels to 1/0. |
| **Hyperparameters** | Settings that control how a model learns. Unlike the model's internal numbers (which are learned automatically), hyperparameters must be set before training. GridSearchCV finds the best ones automatically. |
| **pkl file** | A "pickle" file — Python's way of saving any object (like a trained model) to disk so it can be loaded later without retraining. |
| **Streamlit** | A Python library that lets you build interactive web applications with just Python code — no web development knowledge needed. |
| **Logistic Regression** | The simplest machine learning model for yes/no decisions. Draws a straight line to separate two groups. |
| **Random Forest** | An ensemble of many decision trees. Each tree votes, and the majority wins. |
| **XGBoost** | Gradient boosting — builds trees sequentially, each one correcting the errors of the previous. Very powerful and accurate. |
| **Gradient Boosting** | A technique where models are built one after another, and each new model focuses on fixing what the previous models got wrong. |
| **Decision Tree** | A flowchart-like structure. Asks questions (Is BMI > 30? Is Mammogram Abnormal?) and follows branches until reaching a final decision. |
| **Bootstrapping** | Randomly picking samples from the training data with replacement to train each individual tree in a Random Forest. |
| **Prevalence** | What percentage of the population actually has the disease. If 10% of patients have cancer, the prevalence is 10%. |
| **EDA (Exploratory Data Analysis)** | The initial investigation of a dataset to understand its patterns, missing values, distributions, and relationships before building any model. |
